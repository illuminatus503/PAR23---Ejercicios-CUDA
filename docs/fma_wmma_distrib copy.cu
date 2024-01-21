#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../include/cuda/fma.cuh"
#include "../../include/cuda/kernel_fma.cuh"
#include "../../include/cuda/kernel_cast.cuh"

#include "../../include/utils.h"
#include "../../include/cuda/error.cuh"

double fma_wmma_gpu_distrib(float *D, const float *A, const float *B, const float *C,
                            const int M, const int N, const int K,
                            const int M_split, const int N_split, const int K_split)
{
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;

    float *A_sub, *B_sub, *C_sub, *D_sub;
    int i_size, j_size, k_size;

    int i, j, k;
    int i_sub, j_sub, k_sub;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    if (M_split <= 0)
    {
        perror("M_split is not positive!");
        exit(EXIT_FAILURE);
    }

    if (N_split <= 0)
    {
        perror("N_split is not positive!");
        exit(EXIT_FAILURE);
    }

    if (K_split <= 0)
    {
        perror("K_split is not positive!");
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
    int Nsub = (N + N_split - 1) / N_split;
    int Ksub = (K + K_split - 1) / K_split;

    // ! Reservamos buffers para los fragmentos de matrices
    float *d_C_sub;
    half *d_A_sub, *d_B_sub;
    float *d_A_sub_f32, *d_B_sub_f32;

    gpuErrchk(cudaMalloc((void **)&d_A_sub_f32, Msub * Ksub * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_B_sub_f32, Ksub * Nsub * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_A_sub, Msub * Ksub * sizeof(half)));
    gpuErrchk(cudaMalloc((void **)&d_B_sub, Ksub * Nsub * sizeof(half)));
    gpuErrchk(cudaMalloc((void **)&d_C_sub, Msub * Nsub * sizeof(float)));

    // ! RUN distributed FMA with WMMA operations
    gpuErrchk(cudaEventRecord(start));

    for (i = 0; i < M; i += Msub)
    {
        i_size = (i + Msub > M) ? M - i : Msub;

        for (j = 0; j < N; j += Nsub)
        {
            j_size = (j + Nsub > N) ? N - j : Nsub;

            // Copiamos datos de C a la memoria del device
            gpuErrchk(cudaMalloc((void **)&d_C_sub, i_size * j_size * sizeof(float)));
            gpuErrchk(cudaMemcpy((void *)d_C_sub, (const void *)(C + i * N + j), i_size * j_size * sizeof(float), cudaMemcpyHostToDevice));

            // Agregamos las multiplicaciones de matrices sucesivas
            for (k = 0; k < K; k += Ksub)
            {
                k_size = (k + Ksub > K) ? K - k : Ksub;

                // Apuntar a las submatrices correspondientes de A y B
                A_sub = (float *)&(A[i * K + k]);
                B_sub = (float *)&(Bt[j * K + k]);

                // Enviar fragmentos de A y B_t (half) transpuesta al dispositivo
                gpuErrchk(cudaMemcpy((void *)d_A_sub_f32, (const void *)A_sub, i_size * k_size * sizeof(half), cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy((void *)d_B_sub_f32, (const void *)B_sub, k_size * j_size * sizeof(half), cudaMemcpyHostToDevice));

                // Configurar el tamaño del bloque y la cuadrícula para el kernel de conversión
                dim3 threadsPerBlockConv(256);
                dim3 blocksPerGridConv((i_size * k_size + threadsPerBlockConv.x - 1) / threadsPerBlockConv.x);

                // Lanzar el kernel de conversión para A
                f32_to_f16<<<blocksPerGridConv, threadsPerBlockConv>>>(d_A_sub, d_A_sub_f32, i_size * k_size);
                cudaCheckError();

                // Lanzar el kernel de conversión para B
                f32_to_f16<<<blocksPerGridConv, threadsPerBlockConv>>>(d_B_sub, d_B_sub_f32, k_size * j_size);
                cudaCheckError();

                gpuErrchk(cudaDeviceSynchronize());

                // Realizar la operación FMA en las submatrices
                // Configuración del tamaño de bloque y cuadrícula para el kernel
                dim3 blockDim(WMMA_M, WMMA_N);
                dim3 gridDim((Msub + WMMA_M - 1) / WMMA_M, (Nsub + WMMA_N - 1) / WMMA_N);

                // Lanzamiento del kernel cuda_fma_wmma_ (B traspuesta)
                cuda_fma_wmma_rows<<<gridDim, blockDim>>>(d_C_sub, d_A_sub, d_B_sub, i_size, j_size, k_size, 1.0f, 1.0f);
                gpuErrchk(cudaPeekAtLastError());
                gpuErrchk(cudaDeviceSynchronize());
            }

            // Copiar los resultados de la multiplicación de matrices desde la GPU al host
            gpuErrchk(cudaMemcpy((void *)(D + i * N + j), (const void *)d_C_sub, i_size * j_size * sizeof(float), cudaMemcpyDeviceToHost));
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

    free(Bt);

    // // Calculamos el tamaño de las submatrices
    // int subM = (M + num_streams - 1) / num_streams; // cuantas matrices por filas
    // int subM_padded = (subM + WMMA_M - 1) / WMMA_M * WMMA_M; // Tamaño de submatriz con padding

    // const int N_padded = (N + WMMA_N - 1) / WMMA_N * WMMA_N;
    // const int K_padded = (K + WMMA_K - 1) / WMMA_K * WMMA_K;

    // // Reservar memoria para B una sola vez, ya que no cambia
    // half *d_B;
    // gpuErrchk(cudaMalloc((void **)&d_B, K_padded * N_padded * sizeof(half)));

    // // Copiar y castear B a half con padding
    // half *B_padded = (half *)calloc(K_padded * N_padded, sizeof(half));
    // for (int i = 0; i < K; ++i)
    // {
    //     for (int j = 0; j < N; ++j)
    //     {
    //         B_padded[i * N_padded + j] = __float2half(B[i * N + j]);
    //     }
    // }
    // gpuErrchk(cudaMemcpyAsync(d_B, B_padded, K_padded * N_padded * sizeof(half), cudaMemcpyHostToDevice, streams[0]));
    // free(B_padded); // Liberar memoria del host

    // // Reservar memoria para d_A_sub, d_C_sub y d_C_padded_sub
    // half *d_A_sub;
    // float *d_C_sub, *d_C_padded_sub;
    // gpuErrchk(cudaMalloc((void **)&d_A_sub, subM_padded * K_padded * sizeof(half)));
    // gpuErrchk(cudaMalloc((void **)&d_C_sub, M * N * sizeof(float)));
    // gpuErrchk(cudaMalloc((void **)&d_C_padded_sub, subM_padded * N_padded * sizeof(float)));

    // gpuErrchk(cudaEventRecord(start));
    // for (int s = 0; s < num_streams; ++s)
    // {
    //     int offset_M = s * subM;
    //     int offset_M_padded = s * subM_padded;

    //     // Padding para A_sub y C_sub
    //     half *A_sub_padded = (half *)calloc(subM_padded * K_padded, sizeof(half));
    //     float *C_sub_padded = (float *)calloc(subM_padded * N_padded, sizeof(float));

    //     for (int i = 0; i < subM && (offset_M + i) < M; ++i)
    //     {
    //         for (int j = 0; j < K; ++j)
    //         {
    //             A_sub_padded[i * K_padded + j] = __float2half(A[(offset_M + i) * K + j]);
    //         }
    //         for (int j = 0; j < N; ++j)
    //         {
    //             C_sub_padded[i * N_padded + j] = C[(offset_M + i) * N + j];
    //         }
    //     }

    //     // Copiar datos al dispositivo en el stream actual
    //     gpuErrchk(cudaMemcpyAsync(d_A_sub, A_sub_padded, subM_padded * K_padded * sizeof(half), cudaMemcpyHostToDevice, streams[s]));
    //     gpuErrchk(cudaMemcpyAsync(d_C_padded_sub, C_sub_padded, subM_padded * N_padded * sizeof(float), cudaMemcpyHostToDevice, streams[s]));
    //     free(A_sub_padded); // Liberar memoria del host
    //     free(C_sub_padded); // Liberar memoria del host

    //     // Dimensiones del grid y del bloque para el stream actual
    //     dim3 blockDim(4 * WARP_SIZE, 4);
    //     dim3 gridDim((subM_padded + (WMMA_M * blockDim.x / WARP_SIZE - 1)) / (WMMA_M * blockDim.x / WARP_SIZE),
    //                  (N_padded + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y));

    //     // Lanzar kernel en el stream actual
    //     cuda_fma_wmma<<<gridDim, blockDim, 0, streams[s]>>>(d_C_padded_sub, d_B, d_A_sub, subM_padded, N_padded, K_padded, 1.0f, 1.0f);

    //     // Copiar los resultados de vuelta al host en el stream actual
    //     gpuErrchk(cudaMemcpyAsync(d_C_sub + offset_M * N, d_C_padded_sub, subM * N * sizeof(float), cudaMemcpyDeviceToHost, streams[s]));
    // }

    // gpuErrchk(cudaEventRecord(stop));

    // // Esperar a que todos los streams completen su trabajo
    // for (int i = 0; i < num_streams; ++i)
    //     gpuErrchk(cudaStreamSynchronize(streams[i]));

    // gpuErrchk(cudaEventSynchronize(stop));
    // gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    // // Copiar los datos de C_sub a la matriz D original
    // gpuErrchk(cudaMemcpy(D, d_C_sub, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // // Liberar recursos
    // gpuErrchk(cudaFree(d_A_sub));
    // gpuErrchk(cudaFree(d_B));
    // gpuErrchk(cudaFree(d_C_sub));
    // gpuErrchk(cudaFree(d_C_padded_sub));

    // // Liberar los streams
    // for (int i = 0; i < num_streams; ++i)
    //     gpuErrchk(cudaStreamDestroy(streams[i]));

    return (double)exe_time_ms;
}
