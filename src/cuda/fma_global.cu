#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#include "../../include/cuda/fma.cuh"
#include "../../include/cuda/kernel_fma.cuh"
#include "../../include/cuda/error.cuh"

double fma_gpu_global(float *D, const float *A, const float *B, const float *C,
                      const int M, const int N, const int K)
{
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;

    size_t free_mem, total_mem;
    dim3 gridDim, blockDim;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // ! CUDA set device and check memory availability
    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaMemGetInfo(&free_mem, &total_mem));
    if (free_mem < ((M * K + K * N + M * N) * sizeof(float)))
    {
        fprintf(stderr, "[ERROR] Not enough memory available!\n");
        exit(1);
    }

    // ! CUDA global memory allocation
    // Declaramos las var. de memoria global
    float *d_C;
    float *d_A, *d_B; // const

    // Reservamos mem. global
    gpuErrchk(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

    // Copiamos los datos desde mem. principal
    gpuErrchk(cudaMemcpy((void *)d_A, (const void *)A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_B, (const void *)B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // ! CUDA layout design
    // Block dimension (in threads per dim)
    blockDim.y = WARP_SIZE;
    blockDim.x = WARP_SIZE;

    // Grid dimension (in blocks per dim)
    gridDim.y = (M + blockDim.y - 1) / blockDim.y;
    gridDim.x = (N + blockDim.x - 1) / blockDim.x;

    // ! CUDA launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_gemm_global<<<blockDim, gridDim>>>(d_C, (const float *)d_A, (const float *)d_B, M, N, K, 1.0, 1.0);
    cudaCheckError();

    // ! CUDA copy data to local mem.
    gpuErrchk(cudaMemcpy((void *)D, (const void *)d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    // ! Free CUDA resources
    gpuErrchk(cudaFree((void *)d_A));
    gpuErrchk(cudaFree((void *)d_B));
    gpuErrchk(cudaFree((void *)d_C));

    return (double)exe_time_ms;
}
