#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "superkmeans/common.h"
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/superkmeans.h"

namespace py = pybind11;

using skmeans::Quantization;

template <typename T, int Flags>
void ValidatePyArray(
    const py::array_t<T, Flags>& arr,
    const std::string& name,
    size_t expected_ndim
) {
    auto info = arr.request();
    if (info.ndim != expected_ndim) {
        throw std::runtime_error(
            name + " must be a " + std::to_string(expected_ndim) + "-dimensional array, got " +
            std::to_string(info.ndim)
        );
    }
    if (!(arr.flags() & py::array::c_style)) {
        throw std::runtime_error(name + " must be C-contiguous (row-major)");
    }
}

// Shared body for assign()/quantized_assign()/assign_training_points(): validate the two float
// matrices, invoke the assign-family callable, and copy the uint32 result into a NumPy array.
template <typename Fn>
py::array_t<uint32_t> AssignmentsToArray(
    py::array_t<float> vectors,
    py::array_t<float> centroids,
    Fn&& call
) {
    ValidatePyArray(vectors, "vectors", 2);
    ValidatePyArray(centroids, "centroids", 2);

    auto vectors_info = vectors.request();
    auto centroids_info = centroids.request();

    size_t n_vectors = vectors_info.shape[0];
    size_t n_centroids = centroids_info.shape[0];
    if (vectors_info.shape[1] != centroids_info.shape[1]) {
        throw std::runtime_error("vectors and centroids must have the same dimensionality");
    }

    auto assignments_vec = call(
        static_cast<const float*>(vectors_info.ptr),
        static_cast<const float*>(centroids_info.ptr),
        n_vectors,
        n_centroids
    );

    auto result = py::array_t<uint32_t>(n_vectors);
    std::memcpy(
        result.request().ptr, assignments_vec.data(), assignments_vec.size() * sizeof(uint32_t)
    );
    return result;
}

// Shared body for train(): run clustering and return the centroids as an (n_clusters, d) NumPy
// array.
template <typename KMeans>
py::array_t<float> TrainToArray(
    KMeans& self,
    py::array_t<float> data,
    py::object queries_obj,
    size_t n_queries
) {
    ValidatePyArray(data, "data", 2);
    auto data_info = data.request();
    size_t n = data_info.shape[0];
    size_t d = data_info.shape[1];
    const float* data_ptr = static_cast<const float*>(data_info.ptr);

    const float* queries_ptr = nullptr;
    if (!queries_obj.is_none()) {
        auto queries = queries_obj.cast<py::array_t<float>>();
        ValidatePyArray(queries, "queries", 2);
        auto queries_info = queries.request();
        if (queries_info.shape[1] != static_cast<ssize_t>(d)) {
            throw std::runtime_error("queries must have the same dimensionality as data");
        }
        n_queries = queries_info.shape[0];
        queries_ptr = static_cast<const float*>(queries_info.ptr);
    }

    auto centroids_vec = self.Train(data_ptr, n, queries_ptr, n_queries);

    size_t n_clusters = self.GetNClusters();
    auto result = py::array_t<float>({n_clusters, d});
    std::memcpy(result.request().ptr, centroids_vec.data(), centroids_vec.size() * sizeof(float));
    return result;
}

// Shared body for train_in_place(): rotate `data` in place and return the centroids. The array is
// declared without py::array::forcecast so pybind refuses a dtype/layout conversion instead of
// silently rotating a temporary copy; writeability is checked explicitly since pybind does not.
template <typename KMeans>
py::array_t<float> TrainInPlaceToArray(
    KMeans& self,
    py::array_t<float, py::array::c_style> data,
    py::object queries_obj,
    size_t n_queries
) {
    ValidatePyArray(data, "data", 2);
    if (!data.writeable()) {
        throw std::runtime_error("data must be writeable to train in place");
    }
    auto data_info = data.request();
    size_t n = data_info.shape[0];
    size_t d = data_info.shape[1];
    float* data_ptr = static_cast<float*>(data_info.ptr);

    const float* queries_ptr = nullptr;
    if (!queries_obj.is_none()) {
        auto queries = queries_obj.cast<py::array_t<float>>();
        ValidatePyArray(queries, "queries", 2);
        auto queries_info = queries.request();
        if (queries_info.shape[1] != static_cast<ssize_t>(d)) {
            throw std::runtime_error("queries must have the same dimensionality as data");
        }
        n_queries = queries_info.shape[0];
        queries_ptr = static_cast<const float*>(queries_info.ptr);
    }

    auto centroids_vec = self.TrainInPlace(data_ptr, n, queries_ptr, n_queries);

    size_t n_clusters = self.GetNClusters();
    auto result = py::array_t<float>({n_clusters, d});
    std::memcpy(result.request().ptr, centroids_vec.data(), centroids_vec.size() * sizeof(float));
    return result;
}

// Shared body for rotate()/unrotate(): applies the trained rotation, or its inverse, to a copy of
// `vectors`. The rotation is only known after training.
template <bool INVERSE, typename KMeans>
py::array_t<float> RotateToArray(const KMeans& self, py::array_t<float> vectors) {
    ValidatePyArray(vectors, "vectors", 2);
    const auto* rotator = self.GetState().rotator;
    if (rotator == nullptr) {
        throw std::runtime_error("rotate/unrotate requires a trained model");
    }
    auto info = vectors.request();
    const size_t n = info.shape[0];
    const size_t d = info.shape[1];
    if (d != rotator->num_dimensions) {
        throw std::runtime_error("vectors must have the same dimensionality as the model");
    }

    auto result = py::array_t<float>({n, d});
    const auto* in = static_cast<const float*>(info.ptr);
    auto* out = static_cast<float*>(result.request().ptr);
    if constexpr (INVERSE) {
        rotator->Unrotate(in, out, static_cast<uint32_t>(n));
    } else {
        rotator->Rotate(in, out, static_cast<uint32_t>(n));
    }
    return result;
}

// Global parameters of the fitted quantizer, keyed per quantizer type. Empty for quantizers with
// no global parameters (f32; LVQ4 is per-vector), None before training.
inline py::dict QuantizerParams(const skmeans::F32Quantizer&, size_t) {
    return {};
}

inline py::dict QuantizerParams(const skmeans::LVQ4Quantizer&, size_t) {
    return {};
}

inline py::dict QuantizerParams(const skmeans::SQ8Quantizer& quantizer, size_t) {
    const auto& p = quantizer.GetParams();
    py::dict params;
    params["base"] = p.quantization_base;
    params["scale"] = p.quantization_scale;
    params["inv_scale"] = p.inv_quantization_scale;
    return params;
}

inline py::dict QuantizerParams(const skmeans::RaBitQQuantizer& quantizer, size_t d) {
    const auto p = quantizer.GetParams();
    auto centroid = py::array_t<float>(static_cast<py::ssize_t>(d));
    std::memcpy(centroid.request().ptr, p.centroid, d * sizeof(float));
    py::dict params;
    params["centroid"] = std::move(centroid);
    params["binary_bytes"] = p.binary_bytes;
    return params;
}

template <typename KMeans>
py::object QuantizationParamsDict(const KMeans& self) {
    const auto* quantizer = self.GetQuantizer();
    if (quantizer == nullptr) {
        return py::none();
    }
    return QuantizerParams(*quantizer, self.GetDimensionality());
}

// Zero-copy view over a buffer owned by the estimator. `owner` becomes the array's base, so the
// buffer cannot be freed while the view is alive.
template <typename T>
py::array_t<T> BufferView(py::object owner, const T* data, size_t rows, size_t cols) {
    return py::array_t<T>({rows, cols}, {cols * sizeof(T), sizeof(T)}, data, std::move(owner));
}

template <typename KMeans>
py::object QuantizedDataView(py::object self_obj) {
    const auto& self = self_obj.cast<const KMeans&>();
    const auto* codes = self.GetQuantizedData();
    if (codes == nullptr) {
        return py::none();
    }
    const auto& state = self.GetState();
    return BufferView(std::move(self_obj), codes, state.n_encoded, state.code_size);
}

template <typename KMeans>
py::object SampledIndicesView(py::object self_obj) {
    const auto& self = self_obj.cast<const KMeans&>();
    if (self.sampled_indices == nullptr) {
        return py::none();
    }
    const size_t n_encoded = self.GetState().n_encoded;
    return py::array_t<size_t>(
        {n_encoded}, {sizeof(size_t)}, self.sampled_indices.get(), std::move(self_obj)
    );
}

template <Quantization q>
void BindSuperKMeans(py::module& m, const char* name) {
    using KMeans = skmeans::SuperKMeans<q>;

    py::class_<KMeans>(m, name, "SuperKMeans clustering")
        .def(
            py::init([](size_t n_clusters,
                        size_t dimensionality,
                        const skmeans::SuperKMeansConfig& config) {
                return new KMeans(n_clusters, dimensionality, config);
            }),
            py::arg("n_clusters"),
            py::arg("dimensionality"),
            py::arg("config"),
            "Initialize SuperKMeans with configuration.\n\n"
            "Args:\n"
            "    n_clusters: Number of clusters to create\n"
            "    dimensionality: Number of dimensions in the data\n"
            "    config: Configuration parameters (SuperKMeansConfig)"
        )

        .def(
            py::init([](size_t n_clusters, size_t dimensionality) {
                return new KMeans(n_clusters, dimensionality);
            }),
            py::arg("n_clusters"),
            py::arg("dimensionality"),
            "Initialize SuperKMeans with default configuration.\n\n"
            "Args:\n"
            "    n_clusters: Number of clusters to create\n"
            "    dimensionality: Number of dimensions in the data"
        )

        .def(
            "train",
            [](KMeans& self, py::array_t<float> data, py::object queries_obj, size_t n_queries) {
                return TrainToArray(self, data, queries_obj, n_queries);
            },
            py::arg("data"),
            py::arg("queries") = py::none(),
            py::arg("n_queries") = 0,
            "Run k-means clustering to determine centroids.\n\n"
            "Args:\n"
            "    data: NumPy array of shape (n, d) with dtype float32\n"
            "    queries: Optional NumPy array of query vectors for recall computation\n"
            "    n_queries: Number of query vectors (ignored if queries is provided)\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_clusters, d) containing centroids"
        )

        .def(
            "train_in_place",
            [](KMeans& self,
               py::array_t<float, py::array::c_style> data,
               py::object queries_obj,
               size_t n_queries) {
                return TrainInPlaceToArray(self, data, queries_obj, n_queries);
            },
            py::arg("data"),
            py::arg("queries") = py::none(),
            py::arg("n_queries") = 0,
            "Run k-means clustering, rotating `data` in place instead of allocating a rotated "
            "copy.\n\n"
            "Halves peak memory. `data` is overwritten with its rotated form and is not restored, "
            "so it must be a writeable, C-contiguous float32 array (no dtype conversion is "
            "performed). unrotate_centroids is forced to "
            "False, so data and centroids stay in the same domain.\n\n"
            "Args:\n"
            "    data: NumPy array of shape (n, d) with dtype float32. Overwritten.\n"
            "    queries: Optional NumPy array of query vectors for recall computation\n"
            "    n_queries: Number of query vectors (ignored if queries is provided)\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_clusters, d) containing centroids"
        )

        .def(
            "assign",
            [](KMeans& self, py::array_t<float> vectors, py::array_t<float> centroids) {
                return AssignmentsToArray(
                    vectors,
                    centroids,
                    [&](const float* v, const float* c, size_t nv, size_t nc) {
                        return self.Assign(v, c, nv, nc);
                    }
                );
            },
            py::arg("vectors"),
            py::arg("centroids"),
            "Assign vectors to their nearest centroid using exact float32 brute force.\n\n"
            "Works with any vectors, not just the training data.\n\n"
            "Args:\n"
            "    vectors: NumPy array of shape (n_vectors, d) with dtype float32\n"
            "    centroids: NumPy array of shape (n_centroids, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_vectors,) with dtype uint32 containing cluster indices"
        )

        .def(
            "quantized_assign",
            [](KMeans& self, py::array_t<float> vectors, py::array_t<float> centroids) {
                return AssignmentsToArray(
                    vectors,
                    centroids,
                    [&](const float* v, const float* c, size_t nv, size_t nc) {
                        return self.QuantizedAssign(v, c, nv, nc);
                    }
                );
            },
            py::arg("vectors"),
            py::arg("centroids"),
            "Assign vectors to their nearest centroid using the trained quantizer.\n\n"
            "Encodes the input vectors and centroids with the fitted quantizer, then searches in\n"
            "the quantized domain. Requires train() to have been called first. Works with any\n"
            "vectors, not just the training data.\n\n"
            "Args:\n"
            "    vectors: NumPy array of shape (n_vectors, d) with dtype float32\n"
            "    centroids: NumPy array of shape (n_centroids, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_vectors,) with dtype uint32 containing cluster indices"
        )

        .def(
            "assign_training_points",
            [](KMeans& self, py::array_t<float> vectors, py::array_t<float> centroids) {
                return AssignmentsToArray(
                    vectors,
                    centroids,
                    [&](const float* v, const float* c, size_t nv, size_t nc) {
                        return self.AssignTrainingPoints(v, c, nv, nc);
                    }
                );
            },
            py::arg("vectors"),
            py::arg("centroids"),
            "Fast assignment using trained state. Requires that the vectors are the same\n"
            "as those used in train(). Leverages training assignments (and pruning) for faster\n"
            "assignment than brute force assign().\n\n"
            "Args:\n"
            "    vectors: NumPy array of shape (n_vectors, d) with dtype float32 (must be training "
            "data)\n"
            "    centroids: NumPy array of shape (n_centroids, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_vectors,) with dtype uint32 containing cluster indices"
        )

        .def("get_n_clusters", &KMeans::GetNClusters, "Get the number of clusters.")

        .def(
            "is_trained",
            [](const KMeans& self) { return self.IsTrained(); },
            "Check whether the model has been trained."
        )

        .def_readonly(
            "iteration_stats",
            &KMeans::iteration_stats,
            "List of statistics for each iteration (read-only)"
        )

        .def_property_readonly(
            "state",
            [](const KMeans& self) { return self.GetState(); },
            "How training was carried out (see SuperKMeansState)"
        )

        .def(
            "rotate",
            &RotateToArray<false, KMeans>,
            py::arg("vectors"),
            "Apply the trained rotation, bringing vectors into the domain the model was trained "
            "in.\n\nArgs:\n"
            "    vectors: NumPy array of shape (n, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n, d) with the rotated vectors"
        )

        .def(
            "unrotate",
            &RotateToArray<true, KMeans>,
            py::arg("vectors"),
            "Undo the trained rotation, bringing vectors back to the input domain.\n\n"
            "Args:\n"
            "    vectors: NumPy array of shape (n, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n, d) with the unrotated vectors"
        )

        .def_property_readonly(
            "quantization_params",
            &QuantizationParamsDict<KMeans>,
            "Global parameters of the fitted quantizer as a dict, or None before training. Empty "
            "for schemes without global parameters"
        )

        .def_property_readonly(
            "quantized_data",
            &QuantizedDataView<KMeans>,
            "Zero-copy view of the encoded training vectors, shape (n_encoded, code_size), or "
            "None when there is no separate encoded buffer (f32)"
        )

        .def_property_readonly(
            "sampled_indices",
            &SampledIndicesView<KMeans>,
            "Zero-copy view mapping encoded row i to original row sampled_indices[i], or None "
            "when no sampling was applied"
        )

        .def("__repr__", [name](const KMeans& self) {
            return std::string("<") + name + ": n_clusters=" + std::to_string(self.GetNClusters()) +
                   ", trained=" + (self.IsTrained() ? "True" : "False") + ">";
        });
}

template <Quantization q>
void BindHierarchicalSuperKMeans(py::module& m, const char* name) {
    using KMeans = skmeans::HierarchicalSuperKMeans<q>;

    py::class_<KMeans>(m, name, "Hierarchical SuperKMeans clustering")
        .def(
            py::init([](size_t n_clusters,
                        size_t dimensionality,
                        const skmeans::HierarchicalSuperKMeansConfig& config) {
                return new KMeans(n_clusters, dimensionality, config);
            }),
            py::arg("n_clusters"),
            py::arg("dimensionality"),
            py::arg("config"),
            "Initialize HierarchicalSuperKMeans with configuration.\n\n"
            "Args:\n"
            "    n_clusters: Number of clusters to create\n"
            "    dimensionality: Number of dimensions in the data\n"
            "    config: Configuration parameters (HierarchicalSuperKMeansConfig)"
        )

        .def(
            py::init([](size_t n_clusters, size_t dimensionality) {
                return new KMeans(n_clusters, dimensionality);
            }),
            py::arg("n_clusters"),
            py::arg("dimensionality"),
            "Initialize HierarchicalSuperKMeans with default configuration.\n\n"
            "Args:\n"
            "    n_clusters: Number of clusters to create\n"
            "    dimensionality: Number of dimensions in the data"
        )

        .def(
            "train",
            [](KMeans& self, py::array_t<float> data, py::object queries_obj, size_t n_queries) {
                return TrainToArray(self, data, queries_obj, n_queries);
            },
            py::arg("data"),
            py::arg("queries") = py::none(),
            py::arg("n_queries") = 0,
            "Run hierarchical k-means clustering to determine centroids.\n\n"
            "Args:\n"
            "    data: NumPy array of shape (n, d) with dtype float32\n"
            "    queries: Optional NumPy array of query vectors for recall computation\n"
            "    n_queries: Number of query vectors (ignored if queries is provided)\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_clusters, d) containing centroids"
        )

        .def(
            "train_in_place",
            [](KMeans& self,
               py::array_t<float, py::array::c_style> data,
               py::object queries_obj,
               size_t n_queries) {
                return TrainInPlaceToArray(self, data, queries_obj, n_queries);
            },
            py::arg("data"),
            py::arg("queries") = py::none(),
            py::arg("n_queries") = 0,
            "Run hierarchical k-means clustering, rotating `data` in place instead of allocating a "
            "rotated copy.\n\n"
            "Halves peak memory. `data` is overwritten with its rotated form and is not restored, "
            "so it must be a writeable, C-contiguous float32 array (no dtype conversion is "
            "performed). unrotate_centroids is forced to "
            "False, so data and centroids stay in the same domain.\n\n"
            "Args:\n"
            "    data: NumPy array of shape (n, d) with dtype float32. Overwritten.\n"
            "    queries: Optional NumPy array of query vectors for recall computation\n"
            "    n_queries: Number of query vectors (ignored if queries is provided)\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_clusters, d) containing centroids"
        )

        .def(
            "assign",
            [](KMeans& self, py::array_t<float> vectors, py::array_t<float> centroids) {
                return AssignmentsToArray(
                    vectors,
                    centroids,
                    [&](const float* v, const float* c, size_t nv, size_t nc) {
                        return self.Assign(v, c, nv, nc);
                    }
                );
            },
            py::arg("vectors"),
            py::arg("centroids"),
            "Assign vectors to their nearest centroid using exact float32 brute force.\n\n"
            "Works with any vectors, not just the training data.\n\n"
            "Args:\n"
            "    vectors: NumPy array of shape (n_vectors, d) with dtype float32\n"
            "    centroids: NumPy array of shape (n_centroids, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_vectors,) with dtype uint32 containing cluster indices"
        )

        .def(
            "quantized_assign",
            [](KMeans& self, py::array_t<float> vectors, py::array_t<float> centroids) {
                return AssignmentsToArray(
                    vectors,
                    centroids,
                    [&](const float* v, const float* c, size_t nv, size_t nc) {
                        return self.QuantizedAssign(v, c, nv, nc);
                    }
                );
            },
            py::arg("vectors"),
            py::arg("centroids"),
            "Assign vectors to their nearest centroid using the trained quantizer.\n\n"
            "Encodes the input vectors and centroids with the fitted quantizer, then searches in\n"
            "the quantized domain. Requires train() to have been called first.\n\n"
            "Args:\n"
            "    vectors: NumPy array of shape (n_vectors, d) with dtype float32\n"
            "    centroids: NumPy array of shape (n_centroids, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_vectors,) with dtype uint32 containing cluster indices"
        )

        .def(
            "assign_training_points",
            [](KMeans& self, py::array_t<float> vectors, py::array_t<float> centroids) {
                return AssignmentsToArray(
                    vectors,
                    centroids,
                    [&](const float* v, const float* c, size_t nv, size_t nc) {
                        return self.AssignTrainingPoints(v, c, nv, nc);
                    }
                );
            },
            py::arg("vectors"),
            py::arg("centroids"),
            "Fast assignment using trained state. Requires that the vectors are the same\n"
            "as those used in train(). Leverages training assignments for faster\n"
            "assignment than brute force assign().\n\n"
            "Args:\n"
            "    vectors: NumPy array of shape (n_vectors, d) with dtype float32 (must be training "
            "data)\n"
            "    centroids: NumPy array of shape (n_centroids, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_vectors,) with dtype uint32 containing cluster indices"
        )

        .def("get_n_clusters", &KMeans::GetNClusters, "Get the number of clusters.")

        .def(
            "is_trained",
            [](const KMeans& self) { return self.IsTrained(); },
            "Check whether the model has been trained."
        )

        .def_readonly(
            "iteration_stats",
            &KMeans::iteration_stats,
            "List of statistics for each iteration (read-only)"
        )

        .def_property_readonly(
            "state",
            [](const KMeans& self) { return self.GetState(); },
            "How training was carried out (see SuperKMeansState)"
        )

        .def(
            "rotate",
            &RotateToArray<false, KMeans>,
            py::arg("vectors"),
            "Apply the trained rotation, bringing vectors into the domain the model was trained "
            "in.\n\nArgs:\n"
            "    vectors: NumPy array of shape (n, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n, d) with the rotated vectors"
        )

        .def(
            "unrotate",
            &RotateToArray<true, KMeans>,
            py::arg("vectors"),
            "Undo the trained rotation, bringing vectors back to the input domain.\n\n"
            "Args:\n"
            "    vectors: NumPy array of shape (n, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n, d) with the unrotated vectors"
        )

        .def_property_readonly(
            "quantization_params",
            &QuantizationParamsDict<KMeans>,
            "Global parameters of the fitted quantizer as a dict, or None before training. Empty "
            "for schemes without global parameters"
        )

        .def_property_readonly(
            "quantized_data",
            &QuantizedDataView<KMeans>,
            "Zero-copy view of the encoded training vectors, shape (n_encoded, code_size), or "
            "None when there is no separate encoded buffer (f32)"
        )

        .def_property_readonly(
            "sampled_indices",
            &SampledIndicesView<KMeans>,
            "Zero-copy view mapping encoded row i to original row sampled_indices[i], or None "
            "when no sampling was applied"
        )

        .def_readonly(
            "hierarchical_iteration_stats",
            &KMeans::hierarchical_iteration_stats,
            "Hierarchical iteration statistics (read-only)"
        )

        .def("__repr__", [name](const KMeans& self) {
            return std::string("<") + name + ": n_clusters=" + std::to_string(self.GetNClusters()) +
                   ", trained=" + (self.IsTrained() ? "True" : "False") + ">";
        });
}

PYBIND11_MODULE(_superkmeans, m) {
    m.doc() =
        "A Super fast K-Means library for High-Dimensional vectors on CPUs (x86, ARM) and GPUs";

    py::class_<skmeans::SuperKMeansConfig>(
        m, "SuperKMeansConfig", "Configuration parameters for SuperKMeans clustering."
    )
        .def(py::init<>(), "Default constructor")

        .def_readwrite(
            "iters",
            &skmeans::SuperKMeansConfig::iters,
            "Number of k-means iterations (default: 10)"
        )
        .def_readwrite(
            "sampling_fraction",
            &skmeans::SuperKMeansConfig::sampling_fraction,
            "Fraction of data to sample, 0.0 to 1.0 (default: 0.3)"
        )
        .def_readwrite(
            "max_points_per_cluster",
            &skmeans::SuperKMeansConfig::max_points_per_cluster,
            "Maximum number of points per cluster to sample (default: 256)"
        )
        .def_readwrite(
            "n_threads",
            &skmeans::SuperKMeansConfig::n_threads,
            "Number of CPU threads, 0 = use all available (default: 0)"
        )
        .def_readwrite(
            "seed",
            &skmeans::SuperKMeansConfig::seed,
            "Random seed for reproducibility (default: 42)"
        )
        .def_readwrite(
            "use_blas_only",
            &skmeans::SuperKMeansConfig::use_blas_only,
            "Use GEMM-only computation without pruning (default: False)"
        )
        .def_readwrite(
            "tol",
            &skmeans::SuperKMeansConfig::tol,
            "Tolerance for shift-based early termination (default: 1e-4)"
        )
        .def_readwrite(
            "recall_tol",
            &skmeans::SuperKMeansConfig::recall_tol,
            "Tolerance for recall-based early termination (default: 0.005)"
        )
        .def_readwrite(
            "early_termination",
            &skmeans::SuperKMeansConfig::early_termination,
            "Whether to stop early on convergence (default: True)"
        )
        .def_readwrite(
            "unrotate_centroids",
            &skmeans::SuperKMeansConfig::unrotate_centroids,
            "Whether to map the centroids back to the input domain before returning them "
            "(default: True). Forced to False by train_in_place(), which leaves the data rotated "
            "and so must return centroids in that same domain"
        )
        .def_readwrite(
            "sample_queries",
            &skmeans::SuperKMeansConfig::sample_queries,
            "Whether to sample queries from data (default: False)"
        )
        .def_readwrite(
            "objective_k",
            &skmeans::SuperKMeansConfig::objective_k,
            "Number of nearest neighbors for recall computation (default: 100)"
        )

        .def_readwrite(
            "full_precision_final_centroids",
            &skmeans::SuperKMeansConfig::full_precision_final_centroids,
            "Recompute final centroids from raw float data (default: False)"
        )

        .def_readwrite(
            "verbose",
            &skmeans::SuperKMeansConfig::verbose,
            "Whether to print progress information (default: False)"
        )
        .def_readwrite(
            "angular",
            &skmeans::SuperKMeansConfig::angular,
            "Whether to use spherical k-means (default: False)"
        )

        .def("__repr__", [](const skmeans::SuperKMeansConfig& config) {
            return "<SuperKMeansConfig: iters=" + std::to_string(config.iters) +
                   ", sampling_fraction=" + std::to_string(config.sampling_fraction) +
                   ", n_threads=" + std::to_string(config.n_threads) + ">";
        });

    py::class_<skmeans::SuperKMeansState>(
        m, "SuperKMeansState", "How training was carried out (read-only)."
    )
        .def_readonly("trained", &skmeans::SuperKMeansState::trained, "Whether training completed")
        .def_readonly(
            "trained_in_place",
            &skmeans::SuperKMeansState::trained_in_place,
            "Whether training rotated the caller's buffer instead of a copy"
        )
        .def_readonly(
            "training_data_rotated",
            &skmeans::SuperKMeansState::training_data_rotated,
            "Whether the training data is in the rotated domain"
        )
        .def_readonly(
            "code_size",
            &skmeans::SuperKMeansState::code_size,
            "Elements per encoded vector (bytes for the quantized schemes)"
        )
        .def_readonly(
            "n_encoded", &skmeans::SuperKMeansState::n_encoded, "Number of encoded vectors"
        )
        .def("__repr__", [](const skmeans::SuperKMeansState& state) {
            return std::string("<SuperKMeansState: trained=") + (state.trained ? "True" : "False") +
                   ", trained_in_place=" + (state.trained_in_place ? "True" : "False") +
                   ", training_data_rotated=" + (state.training_data_rotated ? "True" : "False") +
                   ", code_size=" + std::to_string(state.code_size) +
                   ", n_encoded=" + std::to_string(state.n_encoded) + ">";
        });

    py::class_<skmeans::SuperKMeansIterationStats>(
        m,
        "SuperKMeansIterationStats",
        "Statistics for a single iteration of SuperKMeans clustering."
    )
        .def(py::init<>(), "Default constructor")
        .def_readonly(
            "iteration",
            &skmeans::SuperKMeansIterationStats::iteration,
            "Iteration number (1-indexed)"
        )
        .def_readonly(
            "objective",
            &skmeans::SuperKMeansIterationStats::objective,
            "Total clustering cost (WCSS)"
        )
        .def_readonly(
            "shift",
            &skmeans::SuperKMeansIterationStats::shift,
            "Average squared centroid shift from previous iteration"
        )
        .def_readonly(
            "split",
            &skmeans::SuperKMeansIterationStats::split,
            "Number of clusters that were split (empty cluster handling)"
        )
        .def_readonly(
            "recall", &skmeans::SuperKMeansIterationStats::recall, "Recall@k value (0.0 to 1.0)"
        )
        .def_readonly(
            "not_pruned_pct",
            &skmeans::SuperKMeansIterationStats::not_pruned_pct,
            "Percentage of vectors not pruned (-1.0 if not applicable)"
        )
        .def_readonly(
            "partial_d",
            &skmeans::SuperKMeansIterationStats::partial_d,
            "Number of dimensions used for partial distance (d')"
        )
        .def_readonly(
            "is_gemm_only",
            &skmeans::SuperKMeansIterationStats::is_gemm_only,
            "Whether this iteration used GEMM-only computation"
        )
        .def("__repr__", [](const skmeans::SuperKMeansIterationStats& stats) {
            return "<IterationStats: iter=" + std::to_string(stats.iteration) +
                   ", objective=" + std::to_string(stats.objective) +
                   ", shift=" + std::to_string(stats.shift) + ">";
        });

    py::class_<skmeans::HierarchicalSuperKMeansConfig, skmeans::SuperKMeansConfig>(
        m,
        "HierarchicalSuperKMeansConfig",
        "Configuration parameters for Hierarchical SuperKMeans clustering."
    )
        .def(py::init<>(), "Default constructor")

        .def_readwrite(
            "iters_mesoclustering",
            &skmeans::HierarchicalSuperKMeansConfig::iters_mesoclustering,
            "Number of mesoclustering iterations (default: 10)"
        )
        .def_readwrite(
            "iters_fineclustering",
            &skmeans::HierarchicalSuperKMeansConfig::iters_fineclustering,
            "Number of fineclustering iterations (default: 10)"
        )
        .def_readwrite(
            "iters_refinement",
            &skmeans::HierarchicalSuperKMeansConfig::iters_refinement,
            "Number of refinement iterations (default: 2)"
        )

        .def("__repr__", [](const skmeans::HierarchicalSuperKMeansConfig& config) {
            return "<HierarchicalSuperKMeansConfig: iters_meso=" +
                   std::to_string(config.iters_mesoclustering) +
                   ", iters_fine=" + std::to_string(config.iters_fineclustering) +
                   ", iters_refine=" + std::to_string(config.iters_refinement) +
                   ", sampling_fraction=" + std::to_string(config.sampling_fraction) + ">";
        });

    py::class_<skmeans::HierarchicalSuperKMeansIterationStats>(
        m,
        "HierarchicalSuperKMeansIterationStats",
        "Statistics for Hierarchical SuperKMeans clustering."
    )
        .def(py::init<>(), "Default constructor")
        .def_readonly(
            "mesoclustering_iteration_stats",
            &skmeans::HierarchicalSuperKMeansIterationStats::mesoclustering_iteration_stats,
            "Statistics for mesoclustering iterations"
        )
        .def_readonly(
            "fineclustering_iteration_stats",
            &skmeans::HierarchicalSuperKMeansIterationStats::fineclustering_iteration_stats,
            "Statistics for fineclustering iterations"
        )
        .def_readonly(
            "refinement_iteration_stats",
            &skmeans::HierarchicalSuperKMeansIterationStats::refinement_iteration_stats,
            "Statistics for refinement iterations"
        );

    BindSuperKMeans<Quantization::f32>(m, "SuperKMeans");
    BindSuperKMeans<Quantization::sq8>(m, "SuperKMeansSQ8");
    BindSuperKMeans<Quantization::lvq4>(m, "SuperKMeansLVQ4");
    BindSuperKMeans<Quantization::rabitq>(m, "SuperKMeansRabitQ");
    BindHierarchicalSuperKMeans<Quantization::f32>(m, "HierarchicalSuperKMeans");
    BindHierarchicalSuperKMeans<Quantization::sq8>(m, "HierarchicalSuperKMeansSQ8");
    BindHierarchicalSuperKMeans<Quantization::lvq4>(m, "HierarchicalSuperKMeansLVQ4");
    BindHierarchicalSuperKMeans<Quantization::rabitq>(m, "HierarchicalSuperKMeansRabitQ");
}
