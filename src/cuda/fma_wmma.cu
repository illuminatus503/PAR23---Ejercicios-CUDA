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

    // Matrices en el dispositivo.
    half *d_A, *d_B;
    float *d_C, *d_D;

    // Calculamos el espacio de las matrices padded en el dispositivo
    // Calculamos el tamaño de las matrices padded
    int M_padded = (M + WMMA_M - 1) / WMMA_M * WMMA_M;
    int N_padded = (N + WMMA_N - 1) / WMMA_N * WMMA_N;
    int K_padded = (K + WMMA_K - 1) / WMMA_K * WMMA_K;

    const size_t size_A_padded = M_padded * N_padded * sizeof(half);   // Matriz A (half)
    const size_t size_B_padded = N_padded * K_padded * sizeof(half);   // Matriz B (half)
    const size_t size_CD_padded = M_padded * K_padded * sizeof(float); // Matrices C y D (float)

    // Medimos si hay memoria suficiente en el dispositivo
    // TODO múltiples dispositivos: por defecto, 0
    gpu_id = 0;
    cudaSetDevice(gpu_id);
    update_gpu_info(gpu_array, gpu_id); // actualizamos la información sobre mem.
    device_prop = gpu_array->infoGPU[gpu_id];
    printf("[+] Memoria disponible en gpu%zu: %zu MB disponibles de %zu MB\n",
           gpu_id,
           device_prop.free_mem / 1048576,
           device_prop.total_mem / 1048576);

    if (device_prop.free_mem < (size_A_padded + size_B_padded + 2 * size_CD_padded))
    {
        fprintf(stderr,
                "[ERROR] No hay suficiente memoria disponible en gpu%zu: son necesarios %zu MB\n",
                gpu_id,
                (size_A_padded + size_B_padded + 2 * size_CD_padded) / 1048576);
        exit(1);
    }

    // Generamos nuevas matrices padded, que se pasarán al kernel
    half *A_padded = (half *)malloc(size_A_padded);
    memset(A_padded, 0, size_A_padded);

    half *B_padded = (half *)malloc(size_B_padded);
    memset(B_padded, 0, size_B_padded);

    float *C_padded = (float *)malloc(size_CD_padded);
    memset(C_padded, 0, size_CD_padded);

    float *D_padded = (float *)malloc(size_CD_padded);

    // Copiar los datos de A_, B_, C_ a A_padded, B_padded, C_padded, respectivamente
    // Asegurándose de hacer el cast de A_ y B_ a half
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            A_padded[i * N_padded + j] = __float2half(A_[i * N + j]);
        }
    }

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            B_padded[i * K_padded + j] = __float2half(B_[i * K + j]);
        }
    }

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            C_padded[i * K_padded + j] = C_[i * K + j];
        }
    }

    // Incializamos la medición de tiempo mediante eventos
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Reservamos memoria para las matrices del dispositivo
    gpuErrchk(cudaMalloc((void **)&d_A, size_A_padded));
    gpuErrchk(cudaMalloc((void **)&d_B, size_B_padded));
    gpuErrchk(cudaMalloc((void **)&d_C, size_CD_padded));
    gpuErrchk(cudaMalloc((void **)&d_D, size_CD_padded));

    // Cambiar las copias de memoria y la llamada al kernel para usar las matrices padded
    gpuErrchk(cudaMemcpy((void *)d_A, (const void *)A_padded, size_A_padded, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_B, (const void *)B_padded, size_B_padded, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C_padded, size_CD_padded, cudaMemcpyHostToDevice));

    // Set the CUDA layout
    dim3 blockDim(WARP_SIZE * 4, 4);
    // dim3 gridDim((M_padded + blockDim.x - 1) / blockDim.x,
    //              (K_padded + blockDim.y - 1) / blockDim.y);
    dim3 gridDim((M_padded + (WMMA_M * blockDim.x / WARP_SIZE - 1)) / (WMMA_M * blockDim.x / WARP_SIZE),
                 (K_padded + WMMA_K * blockDim.y - 1) / (WMMA_K * blockDim.y));

    // Launch kernel
    printf("[+] Lanzando el kernel en un grid de %u x %u x %u bloques\n",
           gridDim.x, gridDim.y, gridDim.z);
    gpuErrchk(cudaEventRecord(start));
    cuda_fma_wmma<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_D,
                                         M, N, K,
                                         M_padded, N_padded, K_padded);
    cudaCheckError(); // Check error after execution
    gpuErrchk(cudaEventRecord(stop));
    printf("[+] Recuperando datos del dispositivo...\n");

    // Copy data from device array to host array
    gpuErrchk(cudaMemcpy((void *)D_padded, (const void *)d_D, size_CD_padded, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    // Recuperamos los datos a la matriz D original (sin padding)
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            D[i * K + j] = D_padded[i * K_padded + j];
        }
    }

    // Liberamos los recursos del dispositivo
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaFree(d_D));

    // Liberar la memoria de las matrices padded en el host
    free(A_padded);
    free(B_padded);
    free(C_padded);
    free(D_padded);

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
