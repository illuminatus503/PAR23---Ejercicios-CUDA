#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../include/cuda/fma.cuh"
#include "../../include/cuda/kernel_fma.cuh"

#include "../../include/utils.h"
#include "../../include/cuda/error.cuh"

double fma_wmma_gpu_distrib(float *D, const float *A, const float *B, const float *C,
                            const int M, const int N, const int K)
{
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Vamos a calcular el espacio en memoria para la distribución
    const int num_multiprocessors = 36; // del ej. 1
    int rows_per_stream = (num_multiprocessors > 0) ? M / num_multiprocessors : 0;
    int num_streams = (rows_per_stream > 0) ? max(1, min(num_multiprocessors, M / rows_per_stream)) : 0;

    // Crear streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i)
        gpuErrchk(cudaStreamCreate(&streams[i]));

    // Calculamos el espacio de las matrices padded en el dispositivo
    const int M_padded = (M + WMMA_M - 1) / WMMA_M * WMMA_M;
    const int N_padded = (N + WMMA_N - 1) / WMMA_N * WMMA_N;
    const int K_padded = (K + WMMA_K - 1) / WMMA_K * WMMA_K;

    // Generamos nuevas matrices padded, que se pasarán al kernel
    half *A_padded = (half *)calloc(M_padded * N_padded, sizeof(half));
    half *B_padded = (half *)calloc(N_padded * K_padded, sizeof(half));
    float *C_padded = (float *)calloc(M_padded * N_padded, sizeof(float));

    // Copiar los datos de A_, B_, C_ a A_padded, B_padded, C_padded, respectivamente
    // Asegurándose de hacer el cast de A_ y B_ a half
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            A_padded[i * K_padded + j] = __float2half(A[i * K + j]);
        }
    }

    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B_padded[i * N_padded + j] = __float2half(B[i * N + j]);
        }
    }

    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            C_padded[i * N_padded + j] = C[i * N + j];
        }
    }

    // Reservar memoria para B una sola vez, ya que no cambia
    half *d_B_sub;
    gpuErrchk(cudaMalloc((void **)&d_B_sub, K_padded * N_padded * sizeof(half)));
    gpuErrchk(cudaMemcpyAsync(d_B_sub, B_padded, K_padded * N_padded * sizeof(half), cudaMemcpyHostToDevice, streams[0]));

    // Reservar memoria para d_A_sub y d_C_sub
    half *d_A_sub;
    float *d_C_sub;
    int subM = M / num_streams;                                                             // Tamaño de cada submatriz
    gpuErrchk(cudaMalloc((void **)&d_A_sub, subM * K_padded * sizeof(half) * num_streams)); // Reservar memoria para todos los streams
    gpuErrchk(cudaMalloc((void **)&d_C_sub, subM * N_padded * sizeof(float) * num_streams));

    gpuErrchk(cudaEventRecord(start));
    for (int s = 0; s < num_streams; ++s)
    {
        int offset = s * subM * K_padded;         // Desplazamiento para la submatriz actual
        half *d_A_sub_current = d_A_sub + offset; // Puntero al segmento actual
        float *d_C_sub_current = d_C_sub + offset;

        // Copiar datos al dispositivo en el stream actual
        gpuErrchk(cudaMemcpyAsync(d_A_sub_current, A_padded + offset, subM * K_padded * sizeof(half), cudaMemcpyHostToDevice, streams[s]));
        gpuErrchk(cudaMemcpyAsync(d_B_sub, B_padded, K_padded * N_padded * sizeof(half), cudaMemcpyHostToDevice, streams[s]));
        gpuErrchk(cudaMemcpyAsync(d_C_sub_current, C_padded + offset, subM * N_padded * sizeof(float), cudaMemcpyHostToDevice, streams[s]));

        // Dimensiones del grid y del bloque para el stream actual
        dim3 blockDim(4 * WARP_SIZE, 4);
        dim3 gridDim((subM + (WMMA_M * blockDim.x / WARP_SIZE - 1)) / (WMMA_M * blockDim.x / WARP_SIZE),
                     (N_padded + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y));

        // Lanzar kernel en el stream actual
        cuda_fma_wmma<<<gridDim, blockDim, 0, streams[s]>>>(d_C_sub_current, d_B_sub, d_A_sub_current, subM, N_padded, K_padded, 1.0f, 1.0f);

        // Copiar los resultados de vuelta al host en el stream actual
        gpuErrchk(cudaMemcpyAsync(C_padded + offset, d_C_sub_current, subM * N_padded * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
    }

    gpuErrchk(cudaEventRecord(stop));

    // Esperar a que todos los streams completen su trabajo
    for (int i = 0; i < num_streams; ++i)
        gpuErrchk(cudaStreamSynchronize(streams[i]));

    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    // Liberar recursos
    gpuErrchk(cudaFree(d_A_sub));
    gpuErrchk(cudaFree(d_B_sub));
    gpuErrchk(cudaFree(d_C_sub));

    // Liberar los streams
    for (int i = 0; i < num_streams; ++i)
        gpuErrchk(cudaStreamDestroy(streams[i]));

    // Recuperamos los datos a la matriz D original (sin padding)
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            D[i * N + j] = C_padded[i * N_padded + j];
        }
    }

    // Liberar la memoria de las matrices padded en el host
    free(A_padded);
    free(B_padded);
    free(C_padded);

    return (double)exe_time_ms;
}
