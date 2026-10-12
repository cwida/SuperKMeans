#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>

namespace skmeans {

/**
 * @brief Where SuperKMeans' large buffers come from; the caller may inject its own (e.g. one that
 * counts toward a memory limit).
 *
 * Allocate throws on failure and never returns nullptr. Free receives the size passed to
 * Allocate. ParallelFor workers allocate concurrently, so implementations must be thread-safe.
 * An injected allocator is borrowed: it must outlive the objects it is bound to, since every
 * buffer is freed through the allocator it came from.
 *
 * Not every allocation goes through it, only the large buffers (centroids, assignments, sample
 * and scratch buffers, quantizer caches, the rotation matrix). The std::vectors returned by the
 * public API, Eigen's internal temporaries (e.g. the QR that builds the rotation), NumKong's
 * packing buffers and small d-sized or per-worker arrays still use the default heap.
 */
class MemoryAllocator {
  public:
    virtual ~MemoryAllocator() = default;
    virtual void* Allocate(size_t bytes) = 0;
    virtual void Free(void* ptr, size_t bytes) = 0;
};

class VanillaAllocator final : public MemoryAllocator {
  public:
    void* Allocate(size_t bytes) override { return ::operator new[](bytes); }
    void Free(void* ptr, size_t /*bytes*/) override { ::operator delete[](ptr); }
};

template <typename T>
class BufferDeleter {
  public:
    BufferDeleter() = default;
    BufferDeleter(MemoryAllocator* allocator, size_t size) : allocator(allocator), size(size) {}
    void operator()(T* ptr) const { allocator->Free(ptr, size * sizeof(T)); }
    size_t Size() const { return size; }

  private:
    MemoryAllocator* allocator = nullptr;
    size_t size = 0;
};

/// A buffer of `size` trivial T that returns its memory to the allocator it came from, which
/// must outlive it.
template <typename T>
using AllocatedBuffer = std::unique_ptr<T[], BufferDeleter<T>>;

/// Uninitialized, like new T[size].
template <typename T>
AllocatedBuffer<T> AllocateBuffer(MemoryAllocator& allocator, size_t size) {
    static_assert(std::is_trivial<T>::value, "an AllocatedBuffer runs no constructors");
    if (size == 0) {
        return AllocatedBuffer<T>();
    }
    return AllocatedBuffer<T>(
        static_cast<T*>(allocator.Allocate(size * sizeof(T))), BufferDeleter<T>(&allocator, size)
    );
}

/// Zeroed, like std::make_unique<T[]>(size).
template <typename T>
AllocatedBuffer<T> AllocateZeroedBuffer(MemoryAllocator& allocator, size_t size) {
    auto buffer = AllocateBuffer<T>(allocator, size);
    if (size > 0) {
        std::memset(buffer.get(), 0, size * sizeof(T));
    }
    return buffer;
}

/// Grows a scratch buffer to at least `size` elements. Unlike std::vector::resize, a regrown
/// buffer keeps none of the old contents and is uninitialized.
template <typename T>
void EnsureBufferSize(AllocatedBuffer<T>& buffer, MemoryAllocator& allocator, size_t size) {
    if (buffer.get_deleter().Size() < size) {
        buffer = AllocateBuffer<T>(allocator, size);
    }
}

/**
 * @brief Base of the classes whose large buffers come from a bound allocator (borrowed, not
 * owned). Unbound, they use the VanillaAllocator.
 */
class AllocatorHolder {
  public:
    void SetAllocator(MemoryAllocator* allocator) { this->allocator = allocator; }

  protected:
    MemoryAllocator& GetAllocator() const {
        static VanillaAllocator vanilla;
        return allocator != nullptr ? *allocator : vanilla;
    }

  private:
    MemoryAllocator* allocator = nullptr;
};

} // namespace skmeans
