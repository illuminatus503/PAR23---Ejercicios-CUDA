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
                            const int M, const int N, const int K,
                            const int M_split, const int N_split, const int K_split)
{
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;

    int i, j, k;
    int i_size, j_size, k_size;
    int i_size_padded, j_size_padded, k_size_padded;
    int i_frag, j_frag;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    if (M_split <= 0 || N_split <= 0 || K_split <= 0)
    {
        perror("M_split, N_split or K_split is not positive!");
        exit(EXIT_FAILURE);
    }

    // Trasponemos B para matmul (accesos por filas en GPU)
    float *Bt = (float *)malloc(sizeof(float) * K * N);
    for (i = 0; i < K; i++)
    {
        for (j = 0; j < N; j++)
        {
            Bt[j * K + i] = B[i * N + j];
        }
    }

    // Calcular el tamaño de cada submatriz (considerando el padding si es necesario)
    int Msub = (M + M_split - 1) / M_split;
    Msub = (Msub + WMMA_M - 1) / WMMA_M * WMMA_M;
    int Nsub = (N + N_split - 1) / N_split;
    Nsub = (Nsub + WMMA_N - 1) / WMMA_N * WMMA_N;
    int Ksub = (K + K_split - 1) / K_split;
    Ksub = (Ksub + WMMA_K - 1) / WMMA_K * WMMA_K;

    // Reservamos e inicializamos matrices host con padding
    float *A_sub_padded = (float *)calloc(Msub * Ksub, sizeof(float));
    float *B_sub_padded = (float *)calloc(Ksub * Nsub, sizeof(float));
    float *C_sub_padded = (float *)calloc(Msub * Nsub, sizeof(float));

    // Reservamos buffers para los fragmentos de matrices en la GPU
    float *d_C_sub;
    half *d_A_sub, *d_B_sub;
    float *d_A_sub_f32, *d_B_sub_f32;

    gpuErrchk(cudaMalloc((void **)&d_A_sub_f32, Msub * Ksub * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_B_sub_f32, Ksub * Nsub * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_A_sub, Msub * Ksub * sizeof(half)));
    gpuErrchk(cudaMalloc((void **)&d_B_sub, Ksub * Nsub * sizeof(half)));
    gpuErrchk(cudaMalloc((void **)&d_C_sub, Msub * Nsub * sizeof(float)));

    // RUN distributed FMA with WMMA operations
    gpuErrchk(cudaEventRecord(start));

    for (i = 0; i < M; i += Msub)
    {
        // Calculamos la dimensión de filas del fragmento de D (padded)
        i_size = (i + Msub > M) ? M - i : Msub;
        i_size_padded = (i_size + WMMA_M - 1) / WMMA_M * WMMA_M;

        for (j = 0; j < N; j += Nsub)
        {
            // Calculamos la dimensión de columnas del fragmento de D (padded)
            j_size = (j + Nsub > N) ? N - j : Nsub;
            j_size_padded = (j_size + WMMA_N - 1) / WMMA_N * WMMA_N;

            // Inicializamos el fragmento de C_sub con padding
            memset(C_sub_padded, 0, Msub * Nsub * sizeof(float));
            for (i_frag = 0; i_frag < i_size; ++i_frag)
            {
                memcpy(C_sub_padded + i_frag * Nsub,
                       C + (i + i_frag) * N + j,
                       j_size * sizeof(float));
            }

            // Copiamos datos de C_sub_padded a la memoria del dispositivo
            gpuErrchk(cudaMemcpy((void *)d_C_sub,
                                 (const void *)(C_sub_padded + i * Nsub + j),
                                 i_size * j_size * sizeof(float),
                                 cudaMemcpyHostToDevice));

            // Agregamos las multiplicaciones de matrices sucesivas
            for (k = 0; k < K; k += Ksub)
            {
                k_size = (k + Ksub > K) ? K - k : Ksub;
                k_size_padded = (k_size + WMMA_K - 1) / WMMA_K * WMMA_K;

                // Inicializamos los fragmentos de A_sub y B_sub con padding
                memset(A_sub_padded, 0, Msub * Ksub * sizeof(float));
                memset(B_sub_padded, 0, Nsub * Ksub * sizeof(float));
                for (i_frag = 0; i_frag < i_size; i_frag++)
                {
                    memcpy(A_sub_padded + i_frag * Ksub,
                           A + (i + i_frag) * K + k,
                           k_size * sizeof(float));
                }

                for (j_frag = 0; j_frag < j_size; j_frag++)
                {
                    memcpy(B_sub_padded + j_frag * Ksub,
                           Bt + (j + j_frag) * K + k,
                           k_size * sizeof(float));
                }

                // Copiamos y convertimos los fragmentos a half antes de enviarlos a d_A_sub y d_B_sub
                gpuErrchk(cudaMemcpy((void *)d_A_sub_f32,
                                     (const void *)A_sub_padded,
                                     i_size_padded * k_size_padded * sizeof(float),
                                     cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy((void *)d_B_sub_f32,
                                     (const void *)B_sub_padded,
                                     k_size_padded * j_size_padded * sizeof(float),
                                     cudaMemcpyHostToDevice));

                // Convertimos los fragmentos a half antes de enviarlos a d_A_sub y d_B_sub
                f32_to_f16<<<256, (i_size_padded * k_size_padded + 256 - 1) / 256>>>(d_A_sub, d_A_sub_f32, i_size_padded * k_size_padded);
                cudaCheckError();
                f32_to_f16<<<256, (i_size_padded * k_size_padded + 256 - 1) / 256>>>(d_B_sub, d_B_sub_f32, j_size * k_size_padded);
                cudaCheckError();
                gpuErrchk(cudaDeviceSynchronize());

                // Realizamos la operación FMA en las submatrices
                dim3 blockDim(WMMA_M, WMMA_N);
                dim3 gridDim((i_size_padded + WMMA_M - 1) / WMMA_M, (j_size_padded + WMMA_N - 1) / WMMA_N);
                cuda_fma_wmma_rows<<<gridDim, blockDim>>>(d_C_sub, d_A_sub, d_B_sub, i_size_padded, j_size_padded, k_size_padded, 1.0f, 1.0f);
                cudaCheckError();
                gpuErrchk(cudaDeviceSynchronize());
            }

            // Copiamos los resultados de vuelta del dispositivo y extraemos el fragmento sin padding a D
            gpuErrchk(cudaMemcpy(C_sub_padded, d_C_sub, i_size_padded * j_size_padded * sizeof(float), cudaMemcpyDeviceToHost));
            for (i_frag = 0; i_frag < i_size; ++i_frag)
            {
                memcpy(D + (i + i_frag) * N + j,
                       C_sub_padded + i_frag * Nsub,
                       j_size * sizeof(float));
            }
        }
    }

    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    gpuErrchk(cudaFree(d_A_sub));
    gpuErrchk(cudaFree(d_B_sub));
    gpuErrchk(cudaFree(d_A_sub_f32));
    gpuErrchk(cudaFree(d_B_sub_f32));
    gpuErrchk(cudaFree(d_C_sub));

    free(A_sub_padded);
    free(B_sub_padded);
    free(C_sub_padded);
    free(Bt);

    return (double)exe_time_ms;
}
