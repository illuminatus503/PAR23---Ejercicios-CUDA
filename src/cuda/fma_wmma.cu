#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../include/cuda/fma.cuh"
#include "../../include/cuda/kernel_fma.cuh"

#include "../../include/utils.h"
#include "../../include/cuda/error.cuh"
#include "../../include/cuda/utils.cuh"

#define WARP_SIZE 32

double __fma_wmma_gpu(float *A_, float *B_, float *C_, float *D,
                      int M, int N, int K,
                      struct info_t *gpu_array)
{
    size_t gpu_id;
    infoGPU_t device_prop;
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;

    // Calculamos el tamaño de las matrices padded
    int M_padded = (M + WMMA_M - 1) / WMMA_M * WMMA_M;
    int N_padded = (N + WMMA_N - 1) / WMMA_N * WMMA_N;
    int K_padded = (K + WMMA_K - 1) / WMMA_K * WMMA_K;

    // Generamos nuevas matrices padded, que se pasarán al kernel
    // Para A (M_padded x K_padded)
    half *A_padded = (half *)malloc(M_padded * K_padded * sizeof(half));
    memset(A_padded, 0, M_padded * K_padded * sizeof(half));

    // Para B (K_padded x N_padded)
    half *B_padded = (half *)malloc(K_padded * N_padded * sizeof(half));
    memset(B_padded, 0, K_padded * N_padded * sizeof(half));

    // Para C (M_padded x N_padded)
    float *C_padded = (float *)malloc(M_padded * N_padded * sizeof(float));
    memset(C_padded, 0, M_padded * N_padded * sizeof(float));

    // Calculamos el espacio necesario en bytes para las matrices
    half *d_A, *d_B, *h_A_half, *h_B_half;
    float *d_C, *d_D;
    const size_t size_A = N * M * sizeof(half);   // Matriz A (half)
    const size_t size_B = M * K * sizeof(half);   // Matriz B (half)
    const size_t size_CD = N * K * sizeof(float); // Matrices C y D (float)

    // TODO múltiples dispositivos: por defecto, 0
    gpu_id = 0;
    cudaSetDevice(gpu_id);
    update_gpu_info(gpu_array, gpu_id); // actualizamos la información sobre mem.
    device_prop = gpu_array->infoGPU[gpu_id];

    // Incializamos la medición de tiempo mediante eventos
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Medimos la memoria disponible, total del dispositivo en este
    // momento:
    printf("[+] Memoria disponible en gpu%zu: %zu MB disponibles de %zu MB\n",
           gpu_id,
           device_prop.free_mem / 1048576,
           device_prop.total_mem / 1048576);

    if (device_prop.free_mem < (size_A + size_B + 2 * size_CD))
    {
        fprintf(stderr,
                "[ERROR] No hay suficiente memoria disponible en gpu%zu: son necesarios %zu MB\n",
                gpu_id,
                (size_A + size_B + 2 * size_CD) / 1048576);
        exit(1);
    }

    // Reservamos memoria para las matrices en el dispositivo
    gpuErrchk(cudaMalloc((void **)&d_A, size_A));
    gpuErrchk(cudaMalloc((void **)&d_B, size_B));
    gpuErrchk(cudaMalloc((void **)&d_C, size_CD));
    gpuErrchk(cudaMalloc((void **)&d_D, size_CD));

    // Copiamos los datos necesarios para la operación: matrices A, B y C
    // Reservamos previamente memoria en half (host) para matmul
    h_A_half = (half *)malloc(size_A);
    h_B_half = (half *)malloc(size_B);
    for (int i = 0; i < N * M; ++i)
    {
        h_A_half[i] = __float2half(A_[i]);
    }
    for (int i = 0; i < M * K; ++i)
    {
        h_B_half[i] = __float2half(B_[i]);
    }

    // Copiamos a device y liberamos recursos de host
    gpuErrchk(cudaMemcpy((void *)d_A, (const void *)h_A_half, size_A, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_B, (const void *)h_B_half, size_B, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C_, size_CD, cudaMemcpyHostToDevice));
    free(h_A_half);
    free(h_B_half);

    // Set the CUDA layout
    dim3 blockDim(4 * WARP_SIZE, 4);
    dim3 gridDim((M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE),
                 (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y));

    // Launch kernel
    printf("[+] Lanzando el kernel en un grid de %u x %u x %u bloques\n",
           gridDim.x, gridDim.y, gridDim.z);
    gpuErrchk(cudaEventRecord(start));
    cuda_fma_wmma<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_D, N, M, K);
    cudaCheckError(); // Check error after execution
    gpuErrchk(cudaEventRecord(stop));
    printf("[+] Recuperando datos del dispositivo...\n");

    // Copy data from device array to host array
    gpuErrchk(cudaMemcpy((void *)D, (const void *)d_D, size_CD, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    /**
     * Free CUDA mem.
     */
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaFree(d_D));

    return (double)exe_time_ms;
}

double fma_wmma_gpu(float *A_, int N1, int M1,
                    float *B_, int N2, int M2,
                    float *C_, int N3, int M3,
                    float *D, int N, int M,
                    struct info_t *gpu_array)
{
    if (!matrix_checkdims(N1, M1, N2, M2, N3, M3, N, M))
    {
        fprintf(stderr,
                "[DimError] La dimensiones de las matrices no coinciden: A(%d x %d) · B(%d x %d) + C(%d x %d) = D(%d x %d)\n",
                N1, M1, N2, M2, N3, M3, N, M);
        return 0.0; // Asum. que el checkeo no añade sobrecostes
    }

    return __fma_wmma_gpu(A_, B_, C_, D, N, M1, M, gpu_array);
}
