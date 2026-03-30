# GPU SuperKMeans

To use the GPU version of SuperKMeans, add the `-DSKMEANS_ENABLE_GPU=ON` flag to your cmake command, and replace `skmeans::SuperKMeans` with `skmeans::GPUSuperKMeans` in the C++ code. An example GPU version of the `ad_hoc_superkmeans.cpp` file is provided in the benchmarks directory (`gpu_ad_hoc_superkmeans.cpp`).

Note: To replicate the benchmarks from the paper, checkout the `legacy-gpu` branch. 

## Docker Instructions

This section explains how to install and compile the GPU version of SuperKMeans with the supplied docker environment. Docker is not required, but added for reproducability. The Dockerfile also documents the minimum required environment to compile and use the GPU version of SuperKMeans.

1. Create a Docker image for a repeatable build environment with the correct dependencies (example CPU_TARGET=SKYLAKEX):

```sh
cd docker
make build-gpu-image CPU_TARGET=<your_cpu_target> 
```

2. Open and enter the build environment container:

```sh
make open-gpu-container
```

3. Configure CMake:

```sh
cmake . <all_your_other_flags>  -DSKMEANS_COMPILE_BENCHMARKS=ON -DSKMEANS_ENABLE_GPU=ON
```

4. Compile SuperKMeans:

```sh
make -j 
```

5. Run GPU version of SuperKMeans:

```sh
./benchmarks/gpu_ad_hoc_superkmeans.out <dataset_name> 
```
