#include <cuda.h>
#include <cuda_runtime.h>

#include "../../include/cuda/fma.cuh"
#include "../../include/cuda/kernel_fma.cuh"
#include "../../include/cuda/error.cuh"

double fma_gpu_shared(float *D, const float *A, const float *B, const float *C,
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
    dim3 blockDim(WARP_SIZE, WARP_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_fma_shared<<<gridDim, blockDim>>>(d_C, d_A, d_B, M, N, K, 1.0f, 1.0f);
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
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    return (double)exe_time_ms;
}
