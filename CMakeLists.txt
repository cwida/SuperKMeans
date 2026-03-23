cmake_minimum_required(VERSION 3.26)
project(SuperKMeans)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/extern/findFFTW")

add_compile_options(-fPIC)
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
include(CTest)

set(SKMEANS_COMPILE_TESTS OFF CACHE BOOL "Whether to compile tests")
set(SKMEANS_COMPILE_BENCHMARKS OFF CACHE BOOL "Whether to compile benchmarks")
set(SKMEANS_ENABLE_GPU OFF CACHE BOOL "Whether to use the GPU-based implementation of SuperKMeans")
set(SKMEANS_COMPILE_EXAMPLES ON CACHE BOOL "Whether to compile examples")


# Apple Clang does not bundle OpenMP; point FindOpenMP at Homebrew's libomp.
if(APPLE)
    execute_process(COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(HOMEBREW_LIBOMP_PREFIX)
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include"
            CACHE STRING "" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib"
            CACHE FILEPATH "" FORCE)
    endif()
endif()
find_package(OpenMP REQUIRED COMPONENTS CXX)

list(PREPEND CMAKE_PREFIX_PATH /usr/local)

set(MKL_INTERFACE_FULL "intel_lp64")
find_package(MKL CONFIG QUIET)
if (MKL_FOUND)
    message(STATUS "MKL library found")
    message(STATUS "MKL targets: ${MKL_IMPORTED_TARGETS}")
    get_target_property(mkl_includes MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "MKL includes: ${mkl_includes}")
    add_definitions(-DEIGEN_USE_MKL_ALL)
    set(MKL_COMMON_LIBS MKL::MKL OpenMP::OpenMP_CXX m dl)
    set(BLAS_LINK_LIBRARIES "")
else()
    set(MKL_COMMON_LIBS "")
    message(STATUS "MKL not found. Trying to find a BLAS implementation")

    # On macOS, prefer Apple's Accelerate framework over other BLAS implementations
    if(APPLE)
        set(BLA_VENDOR Apple)
        message(STATUS "macOS detected: prioritizing Apple Accelerate framework")
    endif()

    find_package(BLAS REQUIRED)
    message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
    add_definitions(-DEIGEN_USE_BLAS)
    set(BLAS_LINK_LIBRARIES ${BLAS_LIBRARIES} OpenMP::OpenMP_CXX)
endif()
find_package(FFTW QUIET)
if (FFTW_FLOAT_LIB_FOUND)
    message(STATUS "FFTW (+float capabilities) found " ${FFTW_INCLUDE_DIR})
    add_definitions(-DHAS_FFTW)
    include_directories(${FFTW_INCLUDE_DIRS})
else()
    message(STATUS "FFTW (+float capabilities) not found")
endif()

if(SKMEANS_ENABLE_GPU)
    message(STATUS "GPU enabled")
    add_definitions(-DSKMEANS_ENABLE_GPU)
else()
    message(STATUS "GPU disabled")
endif()

set(XNNPACK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(XNNPACK_BUILD_ALL_MICROKERNELS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    xnnpack
    GIT_REPOSITORY https://github.com/google/XNNPACK.git
    GIT_TAG master
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(xnnpack)
set(XNNPACK_LINK_LIBRARIES XNNPACK pthreadpool)

add_compile_definitions(CMAKE_SOURCE_DIR="${CMAKE_SOURCE_DIR}" BENCHMARK_TIME)
include_directories(include extern/Eigen)

if(SKMEANS_COMPILE_TESTS)
    enable_testing()
    message(STATUS "Tests enabled")
    add_subdirectory(tests)
else()
    message(STATUS "Tests disabled")
endif()

if(SKMEANS_COMPILE_BENCHMARKS)
    message(STATUS "Benchmarks enabled")
    add_subdirectory(benchmarks)
else()
    message(STATUS "Benchmarks disabled")
endif()

if(SKMEANS_COMPILE_EXAMPLES)
    message(STATUS "Compiling examples")
    add_subdirectory(examples)
endif()


# Auto-detect Python bindings: only build if pybind11 is available
find_package(Python COMPONENTS Interpreter Development.Module QUIET)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)

if(Python_FOUND AND pybind11_FOUND)
    message(STATUS "Python bindings enabled")
    message(STATUS "Python executable: ${Python_EXECUTABLE}")
    message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
    message(STATUS "pybind11 found: ${pybind11_VERSION}")

    # Create Python extension module
    pybind11_add_module(_superkmeans
        python/bindings/bindings.cpp
    )

    target_include_directories(_superkmeans PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/extern/Eigen
    )

    if(MKL_FOUND)
        target_link_libraries(_superkmeans PRIVATE ${MKL_COMMON_LIBS})
    else()
        target_link_libraries(_superkmeans PRIVATE ${BLAS_LINK_LIBRARIES})
    endif()

    target_link_libraries(_superkmeans PRIVATE ${XNNPACK_LINK_LIBRARIES})

    if(FFTW_FLOAT_LIB_FOUND)
        target_link_libraries(_superkmeans PRIVATE ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB})
    endif()

    target_compile_options(_superkmeans PRIVATE
        -Wall
        $<$<CONFIG:Release>:-O3 -march=native>
    )

    target_compile_features(_superkmeans PRIVATE cxx_std_17)

    install(TARGETS _superkmeans
        LIBRARY DESTINATION superkmeans
        COMPONENT python
    )
else()
    message(STATUS "Python bindings disabled (pybind11 not found)")
endif()
