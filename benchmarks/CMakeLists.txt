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
set(FAISS_ENABLE_GPU OFF CACHE BOOL "disable gpu" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "disable faiss tests" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "shared libs" FORCE)
set(FAISS_OPT_LEVEL "${FAISS_OPT_LEVEL}" CACHE STRING "CPU optimization level" FORCE)

FetchContent_Declare(
    faiss
    GIT_REPOSITORY https://github.com/facebookresearch/faiss.git
    GIT_TAG        v1.11.0
)
FetchContent_MakeAvailable(faiss)

# Common benchmark dependencies
set(BENCH_COMMON_LIBS superkmeans)
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

# Add benchmarks directory to include path for bench_utils.h
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Helper: add a benchmark executable with CMAKE_SOURCE_DIR define
function(skmeans_add_benchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ${BENCH_COMMON_LIBS})
    target_compile_definitions(${name} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
endfunction()

# Helper: add a FAISS benchmark executable
function(skmeans_add_faiss_benchmark name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE ${FAISS_COMMON_LIBS} ${BENCH_COMMON_LIBS})
    target_compile_definitions(${name} PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
endfunction()

# SuperKMeans benchmarks
skmeans_add_benchmark(end_to_end_superkmeans.out end_to_end/end_to_end_superkmeans.cpp)
skmeans_add_benchmark(end_to_end_hierarchical.out end_to_end/end_to_end_hierarchical.cpp)
skmeans_add_benchmark(varying_k_superkmeans.out varying_k/varying_k_superkmeans.cpp)
skmeans_add_benchmark(varying_k_hierarchical_superkmeans.out varying_k/varying_k_hierarchical_superkmeans.cpp)
skmeans_add_benchmark(early_termination_superkmeans.out early_termination/early_termination_superkmeans.cpp)
skmeans_add_benchmark(sampling_superkmeans.out sampling/sampling_superkmeans.cpp)
skmeans_add_benchmark(sampling_hierarchical_superkmeans.out sampling/sampling_hierarchical_superkmeans.cpp)
skmeans_add_benchmark(pareto_superkmeans.out pareto/pareto_superkmeans.cpp)
skmeans_add_benchmark(pareto_hierarchical_superkmeans.out pareto/pareto_hierarchical_superkmeans.cpp)
skmeans_add_benchmark(ad_hoc_superkmeans.out ad_hoc_superkmeans.cpp)
skmeans_add_benchmark(ad_hoc_hierarchical_superkmeans.out ad_hoc_hierarchical_superkmeans.cpp)
skmeans_add_benchmark(ad_hoc_assign.out ad_hoc_assign.cpp)
skmeans_add_benchmark(ad_hoc_jlt.out ad_hoc_jlt.cpp)
skmeans_add_faiss_benchmark(ad_hoc_pca.out ad_hoc_pca.cpp)
skmeans_add_benchmark(ad_hoc_superkmeans_sq8.out ad_hoc_superkmeans_sq8.cpp)
skmeans_add_benchmark(ad_hoc_superkmeans_sq4.out ad_hoc_superkmeans_sq4.cpp)
skmeans_add_benchmark(ad_hoc_id_estimator.out ad_hoc_id_estimator.cpp)
skmeans_add_faiss_benchmark(accelerators.out accelerators/accelerators.cpp)
skmeans_add_faiss_benchmark(ad_hoc_superkmeans_rabitq.out ad_hoc_superkmeans_rabitq.cpp)
skmeans_add_benchmark(ad_hoc_assign_quantized.out ad_hoc_assign_quantized.cpp)
skmeans_add_benchmark(sweet_pruning_spot_superkmeans.out sweet_pruning_spot_superkmeans.cpp)
skmeans_add_benchmark(microbenchmark_init_positions_array.out microbenchmarks/microbenchmark_init_positions_array.cpp)
skmeans_add_benchmark(microbenchmark_flip_sign.out microbenchmarks/microbenchmark_flip_sign.cpp)
skmeans_add_benchmark(microbenchmark_horizontal_kernels.out microbenchmarks/microbenchmark_horizontal_kernels.cpp)
skmeans_add_benchmark(cohere_bench_superkmeans.out cohere_bench_superkmeans.cpp)

# ── XNNPACK ──
# Temporarily disable shared libs — XNNPACK/cpuinfo lack -fPIC for shared builds
# set(_saved_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
# set(BUILD_SHARED_LIBS OFF)
# set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
# set(XNNPACK_BUILD_ALL_MICROKERNELS OFF CACHE BOOL "" FORCE)
# FetchContent_Declare(
#     xnnpack
#     GIT_REPOSITORY https://github.com/google/XNNPACK.git
#     GIT_TAG master
#     GIT_SHALLOW TRUE
# )
# FetchContent_MakeAvailable(xnnpack)
# set(BUILD_SHARED_LIBS ${_saved_BUILD_SHARED_LIBS})
# set(XNNPACK_LINK_LIBRARIES XNNPACK pthreadpool)

# # ── oneDNN (for GEMM benchmarks) ──
# set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
# set(DNNL_GPU_RUNTIME "NONE" CACHE STRING "" FORCE)
# FetchContent_Declare(
#     onednn
#     URL https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v3.11.2.tar.gz
# )
# FetchContent_MakeAvailable(onednn)

# ── NumKong (for GEMM benchmarks) ──
FetchContent_Declare(
    numkong
    GIT_REPOSITORY https://github.com/ashvardanian/NumKong.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(numkong)

# ── ForkUnion (lightweight thread pool for NumKong) ──
FetchContent_Declare(
    fork_union
    GIT_REPOSITORY https://github.com/ashvardanian/ForkUnion.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(fork_union)

# GEMM microbenchmarks
# add_executable(gemm_u8_xnn.out gemm/u8/xnn.cpp)
# target_link_libraries(gemm_u8_xnn.out PRIVATE ${XNNPACK_LINK_LIBRARIES})
# target_compile_definitions(gemm_u8_xnn.out PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
# target_include_directories(gemm_u8_xnn.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# add_executable(gemm_u8_onednn.out gemm/u8/onednn.cpp)
# target_link_libraries(gemm_u8_onednn.out PRIVATE dnnl OpenMP::OpenMP_CXX)
# target_compile_definitions(gemm_u8_onednn.out PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
# target_include_directories(gemm_u8_onednn.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(gemm_u8_numkong.out gemm/u8/numkong.cpp)
target_link_libraries(gemm_u8_numkong.out PRIVATE numkong fork_union)
target_compile_definitions(gemm_u8_numkong.out PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(gemm_u8_numkong.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gemm_u8_numkong.out PRIVATE -march=native)

add_executable(gemm_u8_l2_numkong.out gemm/u8/l2_numkong.cpp)
target_link_libraries(gemm_u8_l2_numkong.out PRIVATE numkong fork_union)
target_compile_definitions(gemm_u8_l2_numkong.out PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(gemm_u8_l2_numkong.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gemm_u8_l2_numkong.out PRIVATE -march=native)

# f32 GEMM microbenchmarks
add_executable(gemm_f32_blas.out gemm/f32/blas.cpp)
target_link_libraries(gemm_f32_blas.out PRIVATE ${BENCH_COMMON_LIBS})
target_compile_definitions(gemm_f32_blas.out PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(gemm_f32_blas.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# add_executable(gemm_f32_xnn.out gemm/f32/xnn.cpp)
# target_link_libraries(gemm_f32_xnn.out PRIVATE ${XNNPACK_LINK_LIBRARIES})
# target_compile_definitions(gemm_f32_xnn.out PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
# target_include_directories(gemm_f32_xnn.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# add_executable(gemm_f32_onednn.out gemm/f32/onednn.cpp)
# target_link_libraries(gemm_f32_onednn.out PRIVATE dnnl OpenMP::OpenMP_CXX)
# target_compile_definitions(gemm_f32_onednn.out PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
# target_include_directories(gemm_f32_onednn.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(gemm_f32_numkong.out gemm/f32/numkong.cpp)
target_link_libraries(gemm_f32_numkong.out PRIVATE numkong fork_union)
target_compile_definitions(gemm_f32_numkong.out PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(gemm_f32_numkong.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gemm_f32_numkong.out PRIVATE -march=native)

# Binary Hamming microbenchmarks
add_executable(gemm_binary_numkong.out gemm/binary/numkong.cpp)
target_link_libraries(gemm_binary_numkong.out PRIVATE numkong fork_union)
target_compile_definitions(gemm_binary_numkong.out PRIVATE CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(gemm_binary_numkong.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gemm_binary_numkong.out PRIVATE -march=native)

# FAISS benchmarks
skmeans_add_faiss_benchmark(end_to_end_faiss.out end_to_end/end_to_end_faiss.cpp)
skmeans_add_faiss_benchmark(varying_k_faiss.out varying_k/varying_k_faiss.cpp)
skmeans_add_faiss_benchmark(early_termination_faiss.out early_termination/early_termination_faiss.cpp)
skmeans_add_faiss_benchmark(cohere_bench_faiss.out cohere_bench_faiss.cpp)

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
        ad_hoc_jlt.out
        ad_hoc_pca.out
        ad_hoc_superkmeans_sq8.out
        ad_hoc_superkmeans_sq4.out
        ad_hoc_superkmeans_rabitq.out
        ad_hoc_assign_quantized.out
        sweet_pruning_spot_superkmeans.out
        microbenchmark_init_positions_array.out
        microbenchmark_flip_sign.out
        microbenchmark_horizontal_kernels.out
        cohere_bench_superkmeans.out
        cohere_bench_faiss.out
)
