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
    float *d_A, *d_B, *d_C;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Reservamos memoria para las matrices en el dispositivo
    gpuErrchk(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

    // Copiamos los datos necesarios para la operación: matrices A, B y C
    gpuErrchk(cudaMemcpy((void *)d_A, (const void *)A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_B, (const void *)B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Asegúrate de que el número de hilos por bloque no sea mayor que el máximo permitido
    dim3 threadsPerBlock(WARP_SIZE, WARP_SIZE); // 1024 threads per block
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_fma_global<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, M, N, K, 1.0f, 1.0f);
    cudaCheckError();
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array to host array
    gpuErrchk(cudaMemcpy((void *)D, (const void *)d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    // Free CUDA resources
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));

    return (double)exe_time_ms;
}
