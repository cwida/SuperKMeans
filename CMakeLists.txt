cmake_minimum_required(VERSION 3.26)
project(SuperKMeans)

# Default to Release build if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Options ──
set(SKMEANS_PORTABLE OFF CACHE BOOL "Use portable SIMD flags instead of -march=native (for wheel builds)")
set(SKMEANS_MARCH "native" CACHE STRING "Target architecture for -march (e.g. native, haswell, znver4). Empty string disables -march.")
set(SKMEANS_SKIP_FFTW OFF CACHE BOOL "Skip FFTW dependency entirely")
set(SKMEANS_ENABLE_GPU OFF CACHE BOOL "Whether to use the GPU-based implementation of SuperKMeans")
set(SKMEANS_COMPILE_TESTS OFF CACHE BOOL "Whether to compile tests")
set(SKMEANS_COMPILE_BENCHMARKS OFF CACHE BOOL "Whether to compile benchmarks")
set(SKMEANS_COMPILE_EXAMPLES ${PROJECT_IS_TOP_LEVEL} CACHE BOOL "Whether to compile examples")

# Executor
if(EMSCRIPTEN)
    set(_SKMEANS_DEFAULT_EXECUTOR serial)
else()
    set(_SKMEANS_DEFAULT_EXECUTOR forkunion)
endif()
set(SKMEANS_EXECUTOR ${_SKMEANS_DEFAULT_EXECUTOR} CACHE STRING "Default parallel executor: forkunion, openmp or serial")
set_property(CACHE SKMEANS_EXECUTOR PROPERTY STRINGS forkunion openmp serial)
if(NOT SKMEANS_EXECUTOR MATCHES "^(forkunion|openmp|serial)$")
    message(FATAL_ERROR "SKMEANS_EXECUTOR must be forkunion, openmp or serial (got '${SKMEANS_EXECUTOR}')")
endif()

# GEMM backend
set(SKMEANS_GEMM auto CACHE STRING "GEMM backend: auto (Accelerate on Apple, Eigen elsewhere), eigen, accelerate or blas")
set_property(CACHE SKMEANS_GEMM PROPERTY STRINGS auto eigen accelerate blas)
if(NOT SKMEANS_GEMM MATCHES "^(auto|eigen|accelerate|blas)$")
    message(FATAL_ERROR "SKMEANS_GEMM must be auto, eigen, accelerate or blas (got '${SKMEANS_GEMM}')")
endif()

# ── Compiler flags (top-level only) ──
if(PROJECT_IS_TOP_LEVEL)
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    endif()
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fassociative-math -funroll-loops")
    endif()
endif()

message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CXX flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")

# ── CMake modules ──
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchContent)
include(CheckCXXCompilerFlag)
include(CMakePrintHelpers)
include(CTest)

# ── Accumulator variables for the INTERFACE target ──
set(_SKMEANS_LINK_LIBS "")
set(_SKMEANS_COMPILE_DEFS "")
set(_SKMEANS_INCLUDE_DIRS "")

# ── Executor ──
if(SKMEANS_EXECUTOR STREQUAL "openmp")
    find_package(OpenMP QUIET)
    if(NOT OpenMP_CXX_FOUND)
        if(APPLE)
            message(FATAL_ERROR
                "OpenMP not found (SKMEANS_EXECUTOR=openmp). On macOS, install it with: brew install libomp\n"
                "Then re-run CMake.")
        else()
            message(FATAL_ERROR "OpenMP not found (SKMEANS_EXECUTOR=openmp).")
        endif()
    endif()
    list(APPEND _SKMEANS_LINK_LIBS OpenMP::OpenMP_CXX)
    list(APPEND _SKMEANS_COMPILE_DEFS SKMEANS_EXECUTOR_OPENMP)
elseif(SKMEANS_EXECUTOR STREQUAL "forkunion")
    FetchContent_Declare(forkunion
        GIT_REPOSITORY https://github.com/ashvardanian/ForkUnion.git
        GIT_TAG v3.0.4
        GIT_SHALLOW TRUE
        SOURCE_SUBDIR headers_only
    )
    FetchContent_MakeAvailable(forkunion)
    find_package(Threads REQUIRED)
    list(APPEND _SKMEANS_INCLUDE_DIRS ${forkunion_SOURCE_DIR}/include)
    list(APPEND _SKMEANS_LINK_LIBS Threads::Threads)
    list(APPEND _SKMEANS_COMPILE_DEFS SKMEANS_EXECUTOR_FORKUNION)
endif()
message(STATUS "Executor: ${SKMEANS_EXECUTOR}")

# ── GEMM backend ──
set(_SKMEANS_GEMM ${SKMEANS_GEMM})
if(_SKMEANS_GEMM STREQUAL "auto")
    if(APPLE)
        set(_SKMEANS_GEMM accelerate)
    else()
        set(_SKMEANS_GEMM eigen)
    endif()
endif()

if(_SKMEANS_GEMM STREQUAL "eigen")
    list(APPEND _SKMEANS_COMPILE_DEFS SKMEANS_GEMM_EIGEN)
elseif(_SKMEANS_GEMM STREQUAL "accelerate")
    if(NOT APPLE)
        message(FATAL_ERROR "SKMEANS_GEMM=accelerate is only available on Apple platforms")
    endif()
    set(BLA_VENDOR Apple)
    find_package(BLAS REQUIRED)
    list(APPEND _SKMEANS_LINK_LIBS ${BLAS_LIBRARIES})
    list(APPEND _SKMEANS_COMPILE_DEFS SKMEANS_GEMM_ACCELERATE)
else()
    # Workers call sgemm_ concurrently: the BLAS must be sequential and thread-safe
    list(APPEND _SKMEANS_COMPILE_DEFS SKMEANS_GEMM_BLAS)
    list(PREPEND CMAKE_PREFIX_PATH /usr/local)

    set(MKL_INTERFACE_FULL "intel_lp64")
    set(MKL_THREADING "sequential")
    find_package(MKL CONFIG QUIET)
    if(MKL_FOUND)
        message(STATUS "MKL library found")
        message(STATUS "MKL targets: ${MKL_IMPORTED_TARGETS}")
        get_target_property(mkl_includes MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
        message(STATUS "MKL includes: ${mkl_includes}")
        list(APPEND _SKMEANS_LINK_LIBS MKL::MKL m dl)
    else()
        message(STATUS "MKL not found. Trying to find a BLAS implementation")
        find_package(BLAS QUIET)
        if(NOT BLAS_FOUND)
            message(STATUS "System BLAS not found. Building OpenBLAS from source via FetchContent...")
            FetchContent_Declare(openblas
                URL https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.31/OpenBLAS-0.3.31.tar.gz
            )
            set(BUILD_WITHOUT_LAPACK ON CACHE BOOL "" FORCE)
            set(C_LAPACK OFF CACHE BOOL "" FORCE)
            set(USE_OPENMP OFF CACHE BOOL "" FORCE)
            set(USE_THREAD OFF CACHE BOOL "" FORCE)
            set(USE_LOCKING ON CACHE BOOL "" FORCE)
            set(DYNAMIC_ARCH ON CACHE BOOL "" FORCE)
            set(NOFORTRAN ON CACHE BOOL "" FORCE)
            FetchContent_MakeAvailable(openblas)
            list(APPEND _SKMEANS_LINK_LIBS openblas)
        else()
            message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
            list(APPEND _SKMEANS_LINK_LIBS ${BLAS_LIBRARIES})
        endif()
    endif()
endif()
message(STATUS "GEMM backend: ${_SKMEANS_GEMM}")

# ── FFTW (optional) ──
if(NOT SKMEANS_SKIP_FFTW)
    find_package(FFTW QUIET)
    if(FFTW_FLOAT_LIB_FOUND)
        message(STATUS "FFTW (+float capabilities) found: ${FFTW_INCLUDE_DIRS}")
        list(APPEND _SKMEANS_COMPILE_DEFS HAS_FFTW)
        list(APPEND _SKMEANS_LINK_LIBS ${FFTW_FLOAT_LIB})
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

# ── NumKong (for 8- and 4-bit GEMM) ──
FetchContent_Declare(
    numkong
    GIT_REPOSITORY https://github.com/ashvardanian/NumKong.git
    GIT_TAG v7.8.3
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(numkong)

# ── Ruy (for 8- and 4-bit GEMM in thinner vectors) ──
# Not in Wasm builds: NumKong runs every 8-bit GEMM there. cpuinfo and ruy have no release tags.
if(NOT EMSCRIPTEN)
    if(NOT TARGET cpuinfo::cpuinfo)
        FetchContent_Declare(cpuinfo
            GIT_REPOSITORY https://github.com/pytorch/cpuinfo.git
            GIT_TAG 66ee79c038d70dad9f08705b2c9b3e58f6d8f512
        )
        set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
        set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
        set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)
        set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(cpuinfo)
    endif()
    FetchContent_Declare(ruy
        GIT_REPOSITORY https://github.com/google/ruy.git
        GIT_TAG 2264753777198e4393fb83c44c693462d57a2be1
    )
    set(RUY_MINIMAL_BUILD ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ruy)
    list(APPEND _SKMEANS_LINK_LIBS ruy::ruy cpuinfo::cpuinfo)
endif()

# ── INTERFACE library target ──
add_library(superkmeans INTERFACE)
add_library(superkmeans::superkmeans ALIAS superkmeans)

# Suppress warnings from our headers when used as a dependency
if(NOT PROJECT_IS_TOP_LEVEL)
    set(_SKMEANS_SYSTEM SYSTEM)
endif()

# ── Eigen ──
# Use consumer-provided or system Eigen when available. Fetch Eigen only when
# nothing else is available.
if(TARGET Eigen3::Eigen)
    message(STATUS "Eigen: using existing Eigen3::Eigen target")
else()
    find_package(Eigen3 QUIET)
    if(Eigen3_FOUND)
        message(STATUS "Eigen: using system Eigen3 (${Eigen3_VERSION})")
    else()
        message(STATUS "Eigen: fetching Eigen 5.0.0")
        FetchContent_Declare(eigen
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG 5.0.0
            GIT_SHALLOW TRUE
            SYSTEM
        )
        set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(eigen)
    endif()
endif()
list(APPEND _SKMEANS_LINK_LIBS Eigen3::Eigen)
list(APPEND _SKMEANS_COMPILE_DEFS EIGEN_DONT_PARALLELIZE)
if(EMSCRIPTEN)
    # Eigen's QR overflows the default Wasm stack with larger on-stack temporaries
    list(APPEND _SKMEANS_COMPILE_DEFS EIGEN_STACK_ALLOCATION_LIMIT=16384)
endif()

target_include_directories(superkmeans ${_SKMEANS_SYSTEM} INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${_SKMEANS_INCLUDE_DIRS}
)

target_link_libraries(superkmeans INTERFACE ${_SKMEANS_LINK_LIBS} numkong)
target_compile_definitions(superkmeans INTERFACE ${_SKMEANS_COMPILE_DEFS})
target_compile_features(superkmeans INTERFACE cxx_std_17)

# Architecture flags on the target (propagated to consumers)
if(MSVC)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
        target_compile_options(superkmeans INTERFACE $<$<CONFIG:Release>:/arch:AVX2>)
        # NumKong enables its x86 kernels by MSVC version, not by /arch: keep it to the AVX2 ones
        target_compile_definitions(superkmeans INTERFACE
            NK_TARGET_SKYLAKE=0 NK_TARGET_ICELAKE=0 NK_TARGET_GENOA=0 NK_TARGET_SAPPHIRE=0
            NK_TARGET_SAPPHIREAMX=0 NK_TARGET_GRANITEAMX=0 NK_TARGET_DIAMOND=0 NK_TARGET_TURIN=0
            NK_TARGET_ALDER=0 NK_TARGET_SIERRA=0
        )
        message(STATUS "MSVC: /arch:AVX2")
    endif()
elseif(EMSCRIPTEN)
    target_compile_options(superkmeans INTERFACE $<$<CONFIG:Release>:-O3 -msimd128>)
    message(STATUS "Emscripten: -O3 -msimd128")
elseif(SKMEANS_PORTABLE)
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
