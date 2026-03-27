#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "superkmeans/common.h"
#include "superkmeans/hierarchical_superkmeans.h"
#include "superkmeans/superkmeans.h"

namespace py = pybind11;

template <typename T>
void ValidatePyArray(py::array_t<T> arr, const std::string& name, size_t expected_ndim) {
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

    py::class_<skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>>(
        m, "SuperKMeans", "SuperKMeans clustering"
    )
        .def(
            py::init([](size_t n_clusters,
                        size_t dimensionality,
                        const skmeans::SuperKMeansConfig& config) {
                return new skmeans::SuperKMeans<
                    skmeans::Quantization::f32,
                    skmeans::DistanceFunction::l2>(n_clusters, dimensionality, config);
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
                return new skmeans::SuperKMeans<
                    skmeans::Quantization::f32,
                    skmeans::DistanceFunction::l2>(n_clusters, dimensionality);
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
            [](skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>&
                   self,
               py::array_t<float> data,
               py::object queries_obj,
               size_t n_queries) {
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
                        throw std::runtime_error("queries must have the same dimensionality as data"
                        );
                    }
                    n_queries = queries_info.shape[0];
                    queries_ptr = static_cast<const float*>(queries_info.ptr);
                }

                auto centroids_vec = self.Train(data_ptr, n, queries_ptr, n_queries);

                size_t n_clusters = self.GetNClusters();
                auto result = py::array_t<float>({n_clusters, d});
                auto result_info = result.request();
                float* result_ptr = static_cast<float*>(result_info.ptr);

                // TODO(@lkuffo, high): I am not a fan of this memcpy. Can we do better?
                std::memcpy(result_ptr, centroids_vec.data(), centroids_vec.size() * sizeof(float));

                return result;
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
            "assign",
            [](skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>&
                   self,
               py::array_t<float> vectors,
               py::array_t<float> centroids) {
                ValidatePyArray(vectors, "vectors", 2);
                ValidatePyArray(centroids, "centroids", 2);

                auto vectors_info = vectors.request();
                auto centroids_info = centroids.request();

                size_t n_vectors = vectors_info.shape[0];
                size_t d_vectors = vectors_info.shape[1];
                size_t n_centroids = centroids_info.shape[0];
                size_t d_centroids = centroids_info.shape[1];

                if (d_vectors != d_centroids) {
                    throw std::runtime_error(
                        "vectors and centroids must have the same dimensionality"
                    );
                }

                const float* vectors_ptr = static_cast<const float*>(vectors_info.ptr);
                const float* centroids_ptr = static_cast<const float*>(centroids_info.ptr);

                auto assignments_vec =
                    self.Assign(vectors_ptr, centroids_ptr, n_vectors, n_centroids);

                auto result = py::array_t<uint32_t>(n_vectors);
                auto result_info = result.request();
                uint32_t* result_ptr = static_cast<uint32_t*>(result_info.ptr);

                std::memcpy(
                    result_ptr, assignments_vec.data(), assignments_vec.size() * sizeof(uint32_t)
                );

                return result;
            },
            py::arg("vectors"),
            py::arg("centroids"),
            "Assign vectors to their nearest centroid using brute force.\n\n"
            "Works with any vectors, not just the training data.\n\n"
            "Args:\n"
            "    vectors: NumPy array of shape (n_vectors, d) with dtype float32\n"
            "    centroids: NumPy array of shape (n_centroids, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_vectors,) with dtype uint32 containing cluster indices"
        )

        .def(
            "assign_training_points",
            [](skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>&
                   self,
               py::array_t<float> vectors,
               py::array_t<float> centroids) {
                ValidatePyArray(vectors, "vectors", 2);
                ValidatePyArray(centroids, "centroids", 2);

                auto vectors_info = vectors.request();
                auto centroids_info = centroids.request();

                size_t n_vectors = vectors_info.shape[0];
                size_t d_vectors = vectors_info.shape[1];
                size_t n_centroids = centroids_info.shape[0];
                size_t d_centroids = centroids_info.shape[1];

                if (d_vectors != d_centroids) {
                    throw std::runtime_error(
                        "vectors and centroids must have the same dimensionality"
                    );
                }

                const float* vectors_ptr = static_cast<const float*>(vectors_info.ptr);
                const float* centroids_ptr = static_cast<const float*>(centroids_info.ptr);

                auto assignments_vec =
                    self.AssignTrainingPoints(vectors_ptr, centroids_ptr, n_vectors, n_centroids);

                auto result = py::array_t<uint32_t>(n_vectors);
                auto result_info = result.request();
                uint32_t* result_ptr = static_cast<uint32_t*>(result_info.ptr);

                std::memcpy(
                    result_ptr, assignments_vec.data(), assignments_vec.size() * sizeof(uint32_t)
                );

                return result;
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

        .def(
            "get_n_clusters",
            &skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>::
                GetNClusters,
            "Get the number of clusters.\n\n"
            "Returns:\n"
            "    Number of clusters"
        )

        .def(
            "is_trained",
            &skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>::
                IsTrained,
            "Check whether the model has been trained.\n\n"
            "Returns:\n"
            "    True if trained, False otherwise"
        )

        .def_readonly(
            "iteration_stats",
            &skmeans::SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>::
                iteration_stats,
            "List of statistics for each iteration (read-only)"
        )

        .def(
            "__repr__",
            [](const skmeans::
                   SuperKMeans<skmeans::Quantization::f32, skmeans::DistanceFunction::l2>& self) {
                return "<SuperKMeans: n_clusters=" + std::to_string(self.GetNClusters()) +
                       ", trained=" + (self.IsTrained() ? "True" : "False") + ">";
            }
        );

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

    py::class_<skmeans::HierarchicalSuperKMeans<
        skmeans::Quantization::f32,
        skmeans::DistanceFunction::l2>>(
        m, "HierarchicalSuperKMeans", "Hierarchical SuperKMeans clustering"
    )
        .def(
            py::init([](size_t n_clusters,
                        size_t dimensionality,
                        const skmeans::HierarchicalSuperKMeansConfig& config) {
                return new skmeans::HierarchicalSuperKMeans<
                    skmeans::Quantization::f32,
                    skmeans::DistanceFunction::l2>(n_clusters, dimensionality, config);
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
                return new skmeans::HierarchicalSuperKMeans<
                    skmeans::Quantization::f32,
                    skmeans::DistanceFunction::l2>(n_clusters, dimensionality);
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
            [](skmeans::HierarchicalSuperKMeans<
                   skmeans::Quantization::f32,
                   skmeans::DistanceFunction::l2>& self,
               py::array_t<float> data,
               py::object queries_obj,
               size_t n_queries) {
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
                        throw std::runtime_error("queries must have the same dimensionality as data"
                        );
                    }
                    n_queries = queries_info.shape[0];
                    queries_ptr = static_cast<const float*>(queries_info.ptr);
                }

                auto centroids_vec = self.Train(data_ptr, n, queries_ptr, n_queries);

                size_t n_clusters = self.GetNClusters();
                auto result = py::array_t<float>({n_clusters, d});
                auto result_info = result.request();
                float* result_ptr = static_cast<float*>(result_info.ptr);

                std::memcpy(result_ptr, centroids_vec.data(), centroids_vec.size() * sizeof(float));

                return result;
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
            "assign",
            [](skmeans::HierarchicalSuperKMeans<
                   skmeans::Quantization::f32,
                   skmeans::DistanceFunction::l2>& self,
               py::array_t<float> vectors,
               py::array_t<float> centroids) {
                ValidatePyArray(vectors, "vectors", 2);
                ValidatePyArray(centroids, "centroids", 2);

                auto vectors_info = vectors.request();
                auto centroids_info = centroids.request();

                size_t n_vectors = vectors_info.shape[0];
                size_t d_vectors = vectors_info.shape[1];
                size_t n_centroids = centroids_info.shape[0];
                size_t d_centroids = centroids_info.shape[1];

                if (d_vectors != d_centroids) {
                    throw std::runtime_error(
                        "vectors and centroids must have the same dimensionality"
                    );
                }

                const float* vectors_ptr = static_cast<const float*>(vectors_info.ptr);
                const float* centroids_ptr = static_cast<const float*>(centroids_info.ptr);

                auto assignments_vec =
                    self.Assign(vectors_ptr, centroids_ptr, n_vectors, n_centroids);

                auto result = py::array_t<uint32_t>(n_vectors);
                auto result_info = result.request();
                uint32_t* result_ptr = static_cast<uint32_t*>(result_info.ptr);

                std::memcpy(
                    result_ptr, assignments_vec.data(), assignments_vec.size() * sizeof(uint32_t)
                );

                return result;
            },
            py::arg("vectors"),
            py::arg("centroids"),
            "Assign vectors to their nearest centroid using brute force.\n\n"
            "Works with any vectors, not just the training data.\n\n"
            "Args:\n"
            "    vectors: NumPy array of shape (n_vectors, d) with dtype float32\n"
            "    centroids: NumPy array of shape (n_centroids, d) with dtype float32\n\n"
            "Returns:\n"
            "    NumPy array of shape (n_vectors,) with dtype uint32 containing cluster indices"
        )

        .def(
            "assign_training_points",
            [](skmeans::HierarchicalSuperKMeans<
                   skmeans::Quantization::f32,
                   skmeans::DistanceFunction::l2>& self,
               py::array_t<float> vectors,
               py::array_t<float> centroids) {
                ValidatePyArray(vectors, "vectors", 2);
                ValidatePyArray(centroids, "centroids", 2);

                auto vectors_info = vectors.request();
                auto centroids_info = centroids.request();

                size_t n_vectors = vectors_info.shape[0];
                size_t d_vectors = vectors_info.shape[1];
                size_t n_centroids = centroids_info.shape[0];
                size_t d_centroids = centroids_info.shape[1];

                if (d_vectors != d_centroids) {
                    throw std::runtime_error(
                        "vectors and centroids must have the same dimensionality"
                    );
                }

                const float* vectors_ptr = static_cast<const float*>(vectors_info.ptr);
                const float* centroids_ptr = static_cast<const float*>(centroids_info.ptr);

                auto assignments_vec =
                    self.AssignTrainingPoints(vectors_ptr, centroids_ptr, n_vectors, n_centroids);

                auto result = py::array_t<uint32_t>(n_vectors);
                auto result_info = result.request();
                uint32_t* result_ptr = static_cast<uint32_t*>(result_info.ptr);

                std::memcpy(
                    result_ptr, assignments_vec.data(), assignments_vec.size() * sizeof(uint32_t)
                );

                return result;
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

        .def(
            "get_n_clusters",
            &skmeans::HierarchicalSuperKMeans<
                skmeans::Quantization::f32,
                skmeans::DistanceFunction::l2>::GetNClusters,
            "Get the number of clusters.\n\n"
            "Returns:\n"
            "    Number of clusters"
        )

        .def(
            "is_trained",
            [](const skmeans::HierarchicalSuperKMeans<
                skmeans::Quantization::f32,
                skmeans::DistanceFunction::l2>& self) { return self.IsTrained(); },
            "Check whether the model has been trained.\n\n"
            "Returns:\n"
            "    True if trained, False otherwise"
        )

        .def_readonly(
            "iteration_stats",
            &skmeans::HierarchicalSuperKMeans<
                skmeans::Quantization::f32,
                skmeans::DistanceFunction::l2>::iteration_stats,
            "List of statistics for each iteration (read-only)"
        )

        .def_readonly(
            "hierarchical_iteration_stats",
            &skmeans::HierarchicalSuperKMeans<
                skmeans::Quantization::f32,
                skmeans::DistanceFunction::l2>::hierarchical_iteration_stats,
            "Hierarchical iteration statistics (read-only)"
        )

        .def(
            "__repr__",
            [](const skmeans::HierarchicalSuperKMeans<
                skmeans::Quantization::f32,
                skmeans::DistanceFunction::l2>& self) {
                return "<HierarchicalSuperKMeans: n_clusters=" +
                       std::to_string(self.GetNClusters()) +
                       ", trained=" + (self.IsTrained() ? "True" : "False") + ">";
            }
        );
}
