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
    float exe_time_ms = 0.0;

#ifdef DEBUG
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
#endif

    if (M_split <= 0 || N_split <= 0 || K_split <= 0)
    {
        perror("M_split, N_split or K_split is not positive!");
        exit(EXIT_FAILURE);
    }

    // Calculate submatrix sizes and pad to multiples of 16 for WMMA.
    int i, j, k;
    int i_size, j_size, k_size;
    int max_i_size = ((M + M_split - 1) / M_split + WMMA_M - 1) / WMMA_M * WMMA_M;
    int max_j_size = ((N + N_split - 1) / N_split + WMMA_N - 1) / WMMA_N * WMMA_N;
    int max_k_size = ((K + K_split - 1) / K_split + WMMA_K - 1) / WMMA_K * WMMA_K;

    // Allocate GPU buffers for submatrix fragments.
    half *d_A_sub_f16, *d_B_sub_f16;
    float *d_A_sub_f32, *d_B_sub_f32, *d_C_sub;
    gpuErrchk(cudaMalloc(&d_A_sub_f32, max_i_size * max_k_size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_B_sub_f32, max_k_size * max_j_size * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_A_sub_f16, max_i_size * max_k_size * sizeof(half)));
    gpuErrchk(cudaMalloc(&d_B_sub_f16, max_k_size * max_j_size * sizeof(half)));
    gpuErrchk(cudaMalloc(&d_C_sub, max_i_size * max_j_size * sizeof(float)));

#ifdef DEBUG
    gpuErrchk(cudaEventRecord(start));
#endif

    for (i = 0; i < M; i += max_i_size)
    {
        i_size = min(max_i_size, M - i);

        for (j = 0; j < N; j += max_j_size)
        {
            j_size = min(max_j_size, N - j);

            gpuErrchk(cudaMemset(d_C_sub, 0, max_i_size * max_j_size * sizeof(float)));
            gpuErrchk(cudaMemcpy2D(d_C_sub, max_j_size * sizeof(float),
                                   C + i * N + j, N * sizeof(float),
                                   j_size * sizeof(float), i_size,
                                   cudaMemcpyHostToDevice));

            for (k = 0; k < K; k += max_k_size)
            {
                k_size = min(max_k_size, K - k);

                gpuErrchk(cudaMemset(d_A_sub_f32, 0, max_i_size * max_k_size * sizeof(float)));
                gpuErrchk(cudaMemset(d_B_sub_f32, 0, max_k_size * max_j_size * sizeof(float)));
                gpuErrchk(cudaMemcpy2D(d_A_sub_f32, max_k_size * sizeof(float),
                                       A + i * K + k, K * sizeof(float),
                                       k_size * sizeof(float), i_size,
                                       cudaMemcpyHostToDevice));
                gpuErrchk(cudaMemcpy2D(d_B_sub_f32, max_j_size * sizeof(float),
                                       B + k * N + j, N * sizeof(float),
                                       j_size * sizeof(float), k_size,
                                       cudaMemcpyHostToDevice));

                // Convert A and B fragments to half precision.
                f32_to_f16<<<256, (max_i_size * max_k_size + 255) / 256>>>(d_A_sub_f16, d_A_sub_f32, max_i_size * max_k_size);
                f32_to_f16<<<256, (max_k_size * max_j_size + 255) / 256>>>(d_B_sub_f16, d_B_sub_f32, max_k_size * max_j_size);
                cudaCheckError();
                gpuErrchk(cudaDeviceSynchronize());

                // Perform FMA operation on submatrices.
                dim3 blockDim(WMMA_M, WMMA_N);
                dim3 gridDim((max_i_size + WMMA_M - 1) / WMMA_M, (max_j_size + WMMA_N - 1) / WMMA_N);
                cuda_fma_wmma<<<gridDim, blockDim>>>(d_C_sub, d_A_sub_f16, d_B_sub_f16,
                                                     max_i_size, max_j_size, max_k_size,
                                                     1.0f, 1.0f);
                cudaCheckError();
                gpuErrchk(cudaDeviceSynchronize());
            }

            // Copy results back to host and extract non-padded fragment to D.
            gpuErrchk(cudaMemcpy2D(D + i * N + j, N * sizeof(float),
                                   d_C_sub, max_j_size * sizeof(float),
                                   j_size * sizeof(float), i_size,
                                   cudaMemcpyDeviceToHost));
        }
    }

#ifdef DEBUG
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
#endif

    // Free CUDA resources.
    gpuErrchk(cudaFree(d_A_sub_f16));
    gpuErrchk(cudaFree(d_B_sub_f16));
    gpuErrchk(cudaFree(d_A_sub_f32));
    gpuErrchk(cudaFree(d_B_sub_f32));
    gpuErrchk(cudaFree(d_C_sub));

    return (double)exe_time_ms;
}
