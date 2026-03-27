cmake_minimum_required(VERSION 3.26)
project(SuperKMeans)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Options ──
set(SKMEANS_PORTABLE OFF CACHE BOOL "Use portable SIMD flags instead of -march=native (for wheel builds)")
set(SKMEANS_MARCH "native" CACHE STRING "Target architecture for -march (e.g. native, haswell, znver4). Empty string disables -march.")
set(SKMEANS_SKIP_FFTW OFF CACHE BOOL "Skip FFTW dependency entirely")
set(SKMEANS_ENABLE_GPU OFF CACHE BOOL "Whether to use the GPU-based implementation of SuperKMeans")
set(SKMEANS_COMPILE_TESTS OFF CACHE BOOL "Whether to compile tests")
set(SKMEANS_COMPILE_BENCHMARKS OFF CACHE BOOL "Whether to compile benchmarks")
set(SKMEANS_COMPILE_EXAMPLES ${PROJECT_IS_TOP_LEVEL} CACHE BOOL "Whether to compile examples")

# ── Compiler flags (top-level only) ──
if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fassociative-math -funroll-loops")
    endif()
endif()

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")

# ── CMake modules ──
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/extern/findFFTW")
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
include(CTest)

# ── Accumulator variables for the INTERFACE target ──
set(_SKMEANS_LINK_LIBS "")
set(_SKMEANS_COMPILE_DEFS "")
set(_SKMEANS_INCLUDE_DIRS "")

# ── OpenMP ──
find_package(OpenMP QUIET)
if(NOT OpenMP_CXX_FOUND)
    if(APPLE)
        message(FATAL_ERROR
            "OpenMP not found. On macOS, install it with: brew install libomp\n"
            "Then re-run CMake.")
    else()
        message(FATAL_ERROR "OpenMP not found.")
    endif()
endif()
list(APPEND _SKMEANS_LINK_LIBS OpenMP::OpenMP_CXX)

# ── BLAS / MKL ──
list(PREPEND CMAKE_PREFIX_PATH /usr/local)

set(MKL_INTERFACE_FULL "intel_lp64")
find_package(MKL CONFIG QUIET)
if(MKL_FOUND)
    message(STATUS "MKL library found")
    message(STATUS "MKL targets: ${MKL_IMPORTED_TARGETS}")
    get_target_property(mkl_includes MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "MKL includes: ${mkl_includes}")
    list(APPEND _SKMEANS_LINK_LIBS MKL::MKL m dl)
else()
    message(STATUS "MKL not found. Trying to find a BLAS implementation")

    if(APPLE)
        set(BLA_VENDOR Apple)
        message(STATUS "macOS detected: prioritizing Apple Accelerate framework")
    endif()

    find_package(BLAS QUIET)
    if(NOT BLAS_FOUND)
        if(APPLE)
            message(FATAL_ERROR "BLAS not found. Apple Accelerate should always be available on macOS.")
        else()
            message(STATUS "System BLAS not found. Building OpenBLAS from source via FetchContent...")
            FetchContent_Declare(openblas
                URL https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.31/OpenBLAS-0.3.31.tar.gz
            )
            set(BUILD_WITHOUT_LAPACK ON CACHE BOOL "" FORCE)
            set(C_LAPACK OFF CACHE BOOL "" FORCE)
            set(USE_OPENMP ON CACHE BOOL "" FORCE)
            set(DYNAMIC_ARCH ON CACHE BOOL "" FORCE)
            set(NUM_THREADS 384 CACHE STRING "" FORCE)
            set(NOFORTRAN ON CACHE BOOL "" FORCE)
            FetchContent_MakeAvailable(openblas)
            list(APPEND _SKMEANS_LINK_LIBS openblas)
        endif()
    else()
        message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
        list(APPEND _SKMEANS_LINK_LIBS ${BLAS_LIBRARIES})
    endif()

endif()

# ── FFTW (optional) ──
if(NOT SKMEANS_SKIP_FFTW)
    find_package(FFTW QUIET)
    if(FFTW_FLOAT_LIB_FOUND)
        message(STATUS "FFTW (+float capabilities) found: ${FFTW_INCLUDE_DIRS}")
        list(APPEND _SKMEANS_COMPILE_DEFS HAS_FFTW)
        list(APPEND _SKMEANS_LINK_LIBS ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB})
        list(APPEND _SKMEANS_INCLUDE_DIRS ${FFTW_INCLUDE_DIRS})
    else()
        message(STATUS "FFTW (+float capabilities) not found")
    endif()
else()
    message(STATUS "FFTW skipped (SKMEANS_SKIP_FFTW=ON)")
endif()

# ── GPU ──
if(SKMEANS_ENABLE_GPU)
    message(STATUS "GPU enabled")
    list(APPEND _SKMEANS_COMPILE_DEFS SKMEANS_ENABLE_GPU)
else()
    message(STATUS "GPU disabled")
endif()

# ── INTERFACE library target ──
add_library(superkmeans INTERFACE)
add_library(superkmeans::superkmeans ALIAS superkmeans)

# Suppress warnings from our headers when used as a dependency
if(NOT PROJECT_IS_TOP_LEVEL)
    set(_SKMEANS_SYSTEM SYSTEM)
endif()

# ── Eigen ──
# Use consumer-provided or system Eigen when available to avoid leaking our
# bundled headers into the consumer's include path. Fall back to bundled
# extern/Eigen only when nothing else is available.
if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen: using existing Eigen3::Eigen target")
    list(APPEND _SKMEANS_LINK_LIBS Eigen3::Eigen)
else()
    find_package(Eigen3 QUIET)
    if(Eigen3_FOUND)
        message(STATUS "Eigen: using system Eigen3 (${Eigen3_VERSION})")
        list(APPEND _SKMEANS_LINK_LIBS Eigen3::Eigen)
    else()
        message(STATUS "Eigen: using bundled extern/Eigen")
        list(APPEND _SKMEANS_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/extern/Eigen)
    endif()
endif()

target_include_directories(superkmeans ${_SKMEANS_SYSTEM} INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${_SKMEANS_INCLUDE_DIRS}
)

target_link_libraries(superkmeans INTERFACE ${_SKMEANS_LINK_LIBS})
target_compile_definitions(superkmeans INTERFACE ${_SKMEANS_COMPILE_DEFS})
target_compile_features(superkmeans INTERFACE cxx_std_17)

# Architecture flags on the target (propagated to consumers)
if(SKMEANS_PORTABLE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
        target_compile_options(superkmeans INTERFACE $<$<CONFIG:Release>:-O3 -mavx2 -mfma>)
        message(STATUS "Portable mode: x86_64, using -mavx2 -mfma")
    else()
        target_compile_options(superkmeans INTERFACE $<$<CONFIG:Release>:-O3>)
        message(STATUS "Portable mode: ${CMAKE_SYSTEM_PROCESSOR}, using generic -O3")
    endif()
elseif(SKMEANS_MARCH)
    target_compile_options(superkmeans INTERFACE $<$<CONFIG:Release>:-O3 -march=${SKMEANS_MARCH}>)
    message(STATUS "Architecture: -march=${SKMEANS_MARCH}")
else()
    target_compile_options(superkmeans INTERFACE $<$<CONFIG:Release>:-O3>)
    message(STATUS "Architecture: no -march flag")
endif()

# ── Subdirectories ──
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

# ── Python bindings ──
find_package(Python COMPONENTS Interpreter Development.Module QUIET)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)

if(Python_FOUND AND pybind11_FOUND)
    message(STATUS "Python bindings enabled")
    message(STATUS "Python executable: ${Python_EXECUTABLE}")
    message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")
    message(STATUS "pybind11 found: ${pybind11_VERSION}")

    pybind11_add_module(_superkmeans python/bindings/bindings.cpp)
    target_link_libraries(_superkmeans PRIVATE superkmeans)

    install(TARGETS _superkmeans
        LIBRARY DESTINATION superkmeans
        COMPONENT python
    )
else()
    message(STATUS "Python bindings disabled (pybind11 not found)")
endif()
