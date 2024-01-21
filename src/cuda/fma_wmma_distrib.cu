#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../include/cuda/fma.cuh"
#include "../../include/cuda/kernel_fma.cuh"
#include "../../include/cuda/kernel_linalg.cuh"
#include "../../include/cuda/error.cuh"

double fma_wmma_gpu_distrib(float *D, float *A, float *B, float *C,
                            const int M, const int N, const int K,
                            const int M_split, const int N_split, const int K_split)
{
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;

    int i, j, k;
    int i_size, j_size, k_size;
    int i_size_padded, j_size_padded, k_size_padded;

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    if (M_split <= 0 || N_split <= 0 || K_split <= 0)
    {
        perror("M_split, N_split or K_split is not positive!");
        exit(EXIT_FAILURE);
    }

    // Calcular el tamaño de cada submatriz (considerando el padding si es necesario)
    // Ajustamos el tamaño de cada fragmento al padding que sea necesario, por dimensión,
    // para que sea múltiplo de 16.
    int Msub = (M + M_split - 1) / M_split;
    int max_i_size = (Msub + WMMA_M - 1) / WMMA_M * WMMA_M;
    int Nsub = (N + N_split - 1) / N_split;
    int max_j_size = (Nsub + WMMA_N - 1) / WMMA_N * WMMA_N;
    int Ksub = (K + K_split - 1) / K_split;
    int max_k_size = (Ksub + WMMA_K - 1) / WMMA_K * WMMA_K;

    // Reservamos buffers para los fragmentos de matrices en la GPU
    half *d_A_sub_f16, *d_B_sub_f16;
    float *d_A_sub_f32, *d_B_sub_f32, *d_C_sub;

    gpuErrchk(cudaMalloc((void **)&d_A_sub_f32, max_i_size * max_k_size * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_B_sub_f32, max_k_size * max_j_size * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_A_sub_f16, max_i_size * max_k_size * sizeof(half)));
    gpuErrchk(cudaMalloc((void **)&d_B_sub_f16, max_k_size * max_j_size * sizeof(half)));
    gpuErrchk(cudaMalloc((void **)&d_C_sub, max_i_size * max_j_size * sizeof(float)));

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

            // Inicializamos el fragmento C: copiamos solo los datos necesarios de C
            gpuErrchk(cudaMemset((void *)d_C_sub, 0, max_i_size * max_j_size * sizeof(float)));
            gpuErrchk(cudaMemcpy((void *)d_C_sub,
                                 (const void *)(C + i * Nsub + j),
                                 i_size * j_size * sizeof(float),
                                 cudaMemcpyHostToDevice));

            for (k = 0; k < K; k += Ksub)
            {
                k_size = (k + Ksub > K) ? K - k : Ksub;
                k_size_padded = (k_size + WMMA_K - 1) / WMMA_K * WMMA_K;

                // Inicializamos los fragmentos A y Bt (con padding)
                gpuErrchk(cudaMemset((void *)d_A_sub_f32, 0, max_i_size * max_k_size * sizeof(float)));
                gpuErrchk(cudaMemset((void *)d_B_sub_f32, 0, max_k_size * max_j_size * sizeof(float)));
                gpuErrchk(cudaMemcpy((void *)d_A_sub_f32,
                                     (const void *)(A + i * Ksub + k),
                                     i_size * k_size * sizeof(float),
                                     cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy((void *)d_B_sub_f32,
                                     (const void *)(B + k * Nsub + j),
                                     k_size * j_size * sizeof(float),
                                     cudaMemcpyHostToDevice));

                // Convertimos los fragmentos A y B a half antes de operar
                f32_to_f16<<<256, (max_i_size * max_k_size + 256 - 1) / 256>>>(d_A_sub_f16, d_A_sub_f32, max_i_size * max_k_size);
                cudaCheckError();
                f32_to_f16<<<256, (max_k_size * max_j_size + 256 - 1) / 256>>>(d_B_sub_f16, d_B_sub_f32, max_k_size * max_j_size);
                cudaCheckError();
                gpuErrchk(cudaDeviceSynchronize());

                // Realizamos la operación FMA en las submatrices
                dim3 blockDim(WMMA_M, WMMA_N);
                dim3 gridDim((i_size_padded + WMMA_M - 1) / WMMA_M,
                             (j_size_padded + WMMA_N - 1) / WMMA_N);
                cuda_fma_wmma<<<gridDim, blockDim>>>(d_C_sub,
                                                     d_A_sub_f16, d_B_sub_f16,
                                                     max_i_size, max_j_size, max_k_size,
                                                     1.0f, 1.0f);
                cudaCheckError();
                gpuErrchk(cudaDeviceSynchronize());
            }

            // Copiamos los resultados de vuelta del dispositivo y extraemos el fragmento sin padding a D
            gpuErrchk(cudaMemcpy((void *)(D + i * Nsub + j),
                                 (const void *)d_C_sub,
                                 i_size * j_size * sizeof(float),
                                 cudaMemcpyDeviceToHost));
        }
    }

    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    // Free CUDA resources
    gpuErrchk(cudaFree(d_A_sub_f16));
    gpuErrchk(cudaFree(d_B_sub_f16));
    gpuErrchk(cudaFree(d_A_sub_f32));
    gpuErrchk(cudaFree(d_B_sub_f32));
    gpuErrchk(cudaFree(d_C_sub));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    return (double)exe_time_ms;
}
