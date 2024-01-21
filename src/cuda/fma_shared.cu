#include <cuda.h>
#include <cuda_runtime.h>

#include "../../include/cuda/fma.cuh"
#include "../../include/cuda/kernel_fma.cuh"
#include "../../include/cuda/error.cuh"

double fma_gpu_shared(float *D, float *A, float *B, float *C,
                      const int M, const int N, const int K)
{
    float *d_A, *d_B, *d_C;
    float exe_time_ms = 0.0;

#ifdef DEBUG
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
#endif

    // Reservamos memoria para las matrices en el dispositivo
    gpuErrchk(cudaMalloc((void **)&d_A, M * K * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_B, K * N * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_C, M * N * sizeof(float)));

    // Copiamos los datos necesarios para la operación: matrices A, B y C
    gpuErrchk(cudaMemcpy((void *)d_A, (const void *)A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_B, (const void *)B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Definimos el layout
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

#ifdef DEBUG
    gpuErrchk(cudaEventRecord(start));
#endif

    // Launch kernel
    cuda_fma_shared<<<gridDim, blockDim>>>(d_C, d_A, d_B, M, N, K, 1.0f, 1.0f);
    cudaCheckError();

#ifdef DEBUG
    gpuErrchk(cudaEventRecord(stop));
#else
    gpuErrchk(cudaDeviceSynchronize());
#endif

    // Copy data from device array to host array
    gpuErrchk(cudaMemcpy((void *)D, (const void *)d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef DEBUG
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
#endif

    // Free CUDA resources
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));

    return (double)exe_time_ms;
}
