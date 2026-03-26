#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "superkmeans/common.h"

namespace skmeans {

namespace gpu {

inline void check_CUDA_error(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(
            stderr, "CUDA Error: %s (%d) at %s:%d\n", cudaGetErrorString(code), code, file, line
        );
        if (abort)
            exit(code);
    }
}

template <typename T>
inline std::size_t compute_buffer_size(const std::size_t x) {
    return x * sizeof(T);
}

template <typename T>
inline std::size_t compute_buffer_size(const std::size_t x, const std::size_t y) {
    return x * y * sizeof(T);
}

#define CUDA_SAFE_CALL(ans) check_CUDA_error((ans), __FILE__, __LINE__)

inline void trigger_gpu_initialization() {
    CUDA_SAFE_CALL(cudaSetDevice(0));
}

class ManagedCudaStream {
  public:
    ManagedCudaStream() { CUDA_SAFE_CALL(cudaStreamCreate(&_stream)); }

    void destroy_stream() {
        if (_stream != nullptr) {
            CUDA_SAFE_CALL(cudaStreamDestroy(_stream));
        }
    }

    ~ManagedCudaStream() { destroy_stream(); }

    ManagedCudaStream(const ManagedCudaStream&) = delete;
    ManagedCudaStream& operator=(const ManagedCudaStream&) = delete;

    ManagedCudaStream& operator=(ManagedCudaStream&& other) noexcept {
        if (this != &other) {
            destroy_stream();
            _stream = other._stream;
            other._stream = nullptr;
        }
        return *this;
    }

    ManagedCudaStream(ManagedCudaStream&& other) noexcept : _stream(other._stream) {
        other._stream = nullptr;
    }

    void synchronize() { CUDA_SAFE_CALL(cudaStreamSynchronize(_stream)); }

    cudaStream_t get() const { return _stream; }

  private:
    cudaStream_t _stream;
};

class StreamPool {
  private:
    std::vector<gpu::ManagedCudaStream> streams;

  public:
    StreamPool() = default;
    StreamPool(const size_t n_streams) : streams(n_streams) {};

    size_t size() const { return streams.size(); }

    void synchronize() {
        for (size_t i{0}; i < size(); ++i) {
            streams[i].synchronize();
        }
    }

    gpu::ManagedCudaStream& operator[](const size_t index) {
        assert(index < size());
        return streams[index];
    }

    const gpu::ManagedCudaStream& operator[](const size_t index) const {
        assert(index < size());
        return streams[index];
    }
};

class ManagedCublasHandle {
  public:
    ManagedCublasHandle(cudaStream_t stream) {
        cublasCreate(&handle);
        cublasSetStream(handle, stream);
    }
    ~ManagedCublasHandle() { cublasDestroy(handle); }

    cublasHandle_t handle;
};

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    DeviceBuffer(std::size_t size, cudaStream_t stream) : _size(size), _stream(stream) {
        CUDA_SAFE_CALL(cudaMalloc(&_dev_ptr, _size));
    }

    void free_buffer() {
        if (_dev_ptr) {
            CUDA_SAFE_CALL(cudaFree(_dev_ptr));
						_dev_ptr = nullptr;
        };
    }

    ~DeviceBuffer() { free_buffer(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            free_buffer();
            _dev_ptr = other._dev_ptr;
            _size = other._size;
            _stream = other._stream;
						other._dev_ptr = nullptr;
						other._size = 0;
        }
        return *this;
    }

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : _dev_ptr(other._dev_ptr), _size(other._size), _stream(other._stream) {
        other._dev_ptr = nullptr;
        other._size = 0;
    }

    bool is_valid() const { return _dev_ptr != nullptr && _size != 0; }

    void copy_to_device(const T* host_ptr) {
        assert(is_valid());
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(get()),
            reinterpret_cast<const void*>(host_ptr),
            _size,
            cudaMemcpyHostToDevice,
            _stream
        ));
    }

    void copy_to_device(const T* host_ptr, const std::size_t size) {
        assert(is_valid());
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(get()),
            reinterpret_cast<const void*>(host_ptr),
            size,
            cudaMemcpyHostToDevice,
            _stream
        ));
    }

    void copy_to_device_at_offset(
        const T* host_ptr,
        const std::size_t size,
        const std::size_t offset
    ) {
        assert(is_valid());
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(get()) + offset,
            reinterpret_cast<const void*>(host_ptr),
            size,
            cudaMemcpyHostToDevice,
            _stream
        ));
    }

    void copy_to_host(T* host_ptr) {
        assert(is_valid());
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(host_ptr),
            reinterpret_cast<const void*>(get()),
            _size,
            cudaMemcpyDeviceToHost,
            _stream
        ));
    }

    void copy_to_host(T* host_ptr, const std::size_t size) {
        assert(is_valid());
        CUDA_SAFE_CALL(cudaMemcpyAsync(
            reinterpret_cast<void*>(host_ptr),
            reinterpret_cast<const void*>(get()),
            size,
            cudaMemcpyDeviceToHost,
            _stream
        ));
    }

    T* get() const { return _dev_ptr; }

  private:
    T* _dev_ptr{nullptr};
    std::size_t _size{0};
    cudaStream_t _stream{nullptr};
};

class BatchedMatrixMultiplier {
    using data_t = skmeans_value_t<Quantization::f32>;

  public:
    BatchedMatrixMultiplier(const cudaStream_t stream) : _cublas_handle(stream) {}

    ~BatchedMatrixMultiplier() {}
    BatchedMatrixMultiplier(const BatchedMatrixMultiplier&) = delete;
    BatchedMatrixMultiplier& operator=(const BatchedMatrixMultiplier&) = delete;
    BatchedMatrixMultiplier& operator=(BatchedMatrixMultiplier&&) = delete;

    BatchedMatrixMultiplier(BatchedMatrixMultiplier&&) = default;

    void multiply(
        const data_t* SKM_RESTRICT batch_x_p,
        const data_t* SKM_RESTRICT batch_y_p,
        const size_t batch_n_x,
        const size_t batch_n_y,
        const size_t d,
        const size_t partial_d,
        float* SKM_RESTRICT all_distances_buf
    ) {
        constexpr float ALPHA = 1.0f;
        constexpr float BETA = 0.0f;

        const int m(static_cast<int>(batch_n_y)); // Rows of result (swapped for row-major)
        const int n(static_cast<int>(batch_n_x)); // Cols of result (swapped for row-major)
        const int k(
            static_cast<int>(partial_d > 0 && partial_d < d ? partial_d : d)
        ); // Inner dimension

        const int lda(static_cast<int>(d)); // Leading dimension of y (row stride in row-major)
        const int ldb(static_cast<int>(d)); // Leading dimension of x (row stride in row-major)
        const int ldc(static_cast<int>(batch_n_y)); // Leading dimension of distances

        cublasSgemm(
            _cublas_handle.handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &ALPHA,
            batch_y_p,
            ldb,
            batch_x_p,
            lda,
            &BETA,
            all_distances_buf,
            ldc
        );
    }

  private:
    ManagedCublasHandle _cublas_handle;
};

template <typename norms_t>
struct StreamBuffers {
    cudaStream_t stream;
    gpu::DeviceBuffer<norms_t> all_distances_buf_dev;
    gpu::BatchedMatrixMultiplier multiplier;

    StreamBuffers(
        const size_t x_batch_size,
        const size_t y_batch_size,
        const size_t d,
        const cudaStream_t stream
    )
        : stream(stream), all_distances_buf_dev(
                              gpu::compute_buffer_size<float>(x_batch_size, y_batch_size),
                              stream
                          ),
          multiplier(stream) {}
};

template <typename norms_t>
static std::vector<StreamBuffers<norms_t>> make_stream_buffers(StreamPool& pool, size_t d) {
    const int32_t n = pool.size();

    std::vector<StreamBuffers<norms_t>> v;
    v.reserve(n);

    for (int32_t i = 0; i < n; ++i) {
        v.emplace_back(X_BATCH_SIZE, Y_BATCH_SIZE, d, pool[i].get());
    }

    return v;
}

template <typename data_t, typename norms_t, typename distance_t>
class GPUDeviceContext {
  private:
  public:
    ManagedCudaStream main_stream;
    StreamPool stream_pool;
    DeviceBuffer<data_t> x;
    DeviceBuffer<data_t> y;
    DeviceBuffer<norms_t> norms_x;
    DeviceBuffer<norms_t> norms_y;
    DeviceBuffer<uint32_t> out_knn;
    DeviceBuffer<distance_t> out_distances;
    DeviceBuffer<size_t> out_not_pruned_counts;
    std::vector<StreamBuffers<norms_t>> stream_buffers;
    bool loaded_x_data = false;

    GPUDeviceContext() = default;
    GPUDeviceContext(const size_t n_x, const size_t n_y, const size_t d, const size_t n_streams)
        : stream_pool(n_streams), x(compute_buffer_size<data_t>(n_x, d), main_stream.get()),
          y(compute_buffer_size<data_t>(n_y, d), main_stream.get()),
          norms_x(compute_buffer_size<norms_t>(n_x), main_stream.get()),
          norms_y(compute_buffer_size<norms_t>(n_y), main_stream.get()),
          out_knn(compute_buffer_size<uint32_t>(n_x), main_stream.get()),
          out_distances(compute_buffer_size<distance_t>(n_x), main_stream.get()),
          out_not_pruned_counts(compute_buffer_size<size_t>(n_x), main_stream.get()),
          stream_buffers(make_stream_buffers<norms_t>(stream_pool, d)) {}
};
} // namespace gpu
} // namespace skmeans
