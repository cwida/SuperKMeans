# Benchmark suite for SuperKMeans

# FAISS optimization level - user can set via -DFAISS_OPT_LEVEL=<level>
# Valid values: generic, avx2, avx512, avx512_spr, sve
set(FAISS_OPT_LEVEL "generic" CACHE STRING "FAISS CPU optimization level")
set_property(CACHE FAISS_OPT_LEVEL PROPERTY STRINGS generic avx2 avx512 avx512_spr sve)
message(STATUS "FAISS optimization level: ${FAISS_OPT_LEVEL}")

# Fetch and build FAISS
if (MKL_FOUND)
    message(STATUS "FAISS will be built with MKL")
    set(FAISS_ENABLE_MKL ON CACHE BOOL "enable mkl" FORCE)
endif()

set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "disable python" FORCE)

if (SKMEANS_ENABLE_GPU)
	set(FAISS_ENABLE_GPU ON CACHE BOOL "enable gpu" FORCE)
	set(FAISS_ENABLE_CUVS ON CACHE BOOL "enable CUVS" FORCE)
else()
	set(FAISS_ENABLE_GPU OFF CACHE BOOL "disable gpu" FORCE)
endif()
set(BUILD_TESTING OFF CACHE BOOL "disable faiss tests" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "shared libs" FORCE)
set(FAISS_OPT_LEVEL "${FAISS_OPT_LEVEL}" CACHE STRING "CPU optimization level" FORCE)

if (SKMEANS_ENABLE_GPU) 
	FetchContent_Declare(
			faiss
			GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
			GIT_TAG        v1.13.2
	)
else()
	FetchContent_Declare(
			faiss
			GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
			GIT_TAG        v1.11.0
	)
endif()
FetchContent_MakeAvailable(faiss)

# Common benchmark dependencies
set(BENCH_COMMON_LIBS ${MKL_COMMON_LIBS} ${BLAS_LINK_LIBRARIES})
set(BENCH_COMMON_SRCS)

if (SKMEANS_ENABLE_GPU)
		set(CUDA_LIBS PRIVATE CUDA::cudart CUDA::cublas)
		list(APPEND BENCH_COMMON_LIBS ${CUDA_LIBS})
		list(APPEND BENCH_COMMON_SRCS ../src/gpu/kernels.cu)
endif()

set(FAISS_COMMON_LIBS faiss)

# Link architecture-specific FAISS library based on FAISS_OPT_LEVEL
if(FAISS_OPT_LEVEL STREQUAL "avx512_spr")
    list(APPEND FAISS_COMMON_LIBS faiss_avx512_spr)
elseif(FAISS_OPT_LEVEL STREQUAL "avx512")
    list(APPEND FAISS_COMMON_LIBS faiss_avx512)
elseif(FAISS_OPT_LEVEL STREQUAL "avx2")
    list(APPEND FAISS_COMMON_LIBS faiss_avx2)
elseif(FAISS_OPT_LEVEL STREQUAL "sve")
    list(APPEND FAISS_COMMON_LIBS faiss_sve)
endif()

if (FFTW_FOUND)
    message(STATUS "Linking FFTW: ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB}")
    list(APPEND BENCH_COMMON_LIBS ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB})
endif()

# Add benchmarks directory to include path for bench_utils.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# SuperKMeans benchmark on real datasets with recall computation
add_executable(end_to_end_superkmeans.out end_to_end/end_to_end_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(end_to_end_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

add_executable(end_to_end_hierarchical.out end_to_end/end_to_end_hierarchical.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(end_to_end_hierarchical.out PRIVATE ${BENCH_COMMON_LIBS})

# FAISS benchmark on real datasets with recall computation
add_executable(end_to_end_faiss.out end_to_end/end_to_end_faiss.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(end_to_end_faiss.out PRIVATE ${FAISS_COMMON_LIBS} ${BENCH_COMMON_LIBS})

# Varying K benchmarks (vary n_clusters instead of dataset)
add_executable(varying_k_superkmeans.out varying_k/varying_k_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(varying_k_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

add_executable(varying_k_faiss.out varying_k/varying_k_faiss.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(varying_k_faiss.out PRIVATE ${FAISS_COMMON_LIBS} ${BENCH_COMMON_LIBS})

add_executable(varying_k_hierarchical_superkmeans.out varying_k/varying_k_hierarchical_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(varying_k_hierarchical_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

# Early termination benchmarks
add_executable(early_termination_superkmeans.out early_termination/early_termination_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(early_termination_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

add_executable(early_termination_faiss.out early_termination/early_termination_faiss.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(early_termination_faiss.out PRIVATE ${FAISS_COMMON_LIBS} ${BENCH_COMMON_LIBS})

# Sampling benchmark
add_executable(sampling_superkmeans.out sampling/sampling_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(sampling_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

add_executable(sampling_hierarchical_superkmeans.out sampling/sampling_hierarchical_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(sampling_hierarchical_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

# Pareto benchmark (grid search)
add_executable(pareto_superkmeans.out pareto/pareto_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(pareto_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

# Pareto hierarchical superkmeans benchmark (grid search over hierarchical hyperparameters)
add_executable(pareto_hierarchical_superkmeans.out pareto/pareto_hierarchical_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(pareto_hierarchical_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

# Ad-hoc benchmark (verbose mode, no CSV output)
add_executable(ad_hoc_superkmeans.out ad_hoc_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(ad_hoc_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

# Ad-hoc hierarchical superkmeans benchmark
add_executable(ad_hoc_hierarchical_superkmeans.out ad_hoc_hierarchical_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(ad_hoc_hierarchical_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

# Ad-hoc assign benchmark (compare use_train_state fast path vs brute force)
add_executable(ad_hoc_assign.out ad_hoc_assign.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(ad_hoc_assign.out PRIVATE ${BENCH_COMMON_LIBS})

# Sweet pruning spot benchmark (grid search over pruning parameters)
add_executable(sweet_pruning_spot_superkmeans.out sweet_pruning_spot_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(sweet_pruning_spot_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

# Microbenchmark for InitPositionsArray SIMD optimization
add_executable(microbenchmark_init_positions_array.out microbenchmarks/microbenchmark_init_positions_array.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(microbenchmark_init_positions_array.out PRIVATE ${BENCH_COMMON_LIBS})

# Microbenchmark for FlipSign SIMD optimization
add_executable(microbenchmark_flip_sign.out microbenchmarks/microbenchmark_flip_sign.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(microbenchmark_flip_sign.out PRIVATE ${BENCH_COMMON_LIBS})

# Microbenchmark for Horizontal L2 distance kernels
add_executable(microbenchmark_horizontal_kernels.out microbenchmarks/microbenchmark_horizontal_kernels.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(microbenchmark_horizontal_kernels.out PRIVATE ${BENCH_COMMON_LIBS})

add_executable(cohere_bench_superkmeans.out cohere_bench_superkmeans.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(cohere_bench_superkmeans.out PRIVATE ${BENCH_COMMON_LIBS})

add_executable(cohere_bench_faiss.out cohere_bench_faiss.cpp ${BENCH_COMMON_SRCS})
target_link_libraries(cohere_bench_faiss.out PRIVATE ${FAISS_COMMON_LIBS} ${BENCH_COMMON_LIBS})

add_custom_target(benchmarks
    DEPENDS
        end_to_end_superkmeans.out
        end_to_end_hierarchical.out
        end_to_end_faiss.out
        varying_k_superkmeans.out
        varying_k_faiss.out
        varying_k_hierarchical_superkmeans.out
        early_termination_superkmeans.out
        early_termination_faiss.out
        sampling_superkmeans.out
        sampling_hierarchical_superkmeans.out
        pareto_superkmeans.out
        pareto_hierarchical_superkmeans.out
        ad_hoc_superkmeans.out
        ad_hoc_hierarchical_superkmeans.out
        ad_hoc_assign.out
        sweet_pruning_spot_superkmeans.out
        microbenchmark_init_positions_array.out
        microbenchmark_flip_sign.out
        microbenchmark_horizontal_kernels.out
        cohere_bench_superkmeans.out
        cohere_bench_faiss.out
)
