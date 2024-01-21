#include <cuda.h>
#include <cuda_runtime.h>

#include "../../include/cuda/error.cuh"
#include "../../include/cuda/kernel_linalg.cuh"
#include "../../include/cuda/linalg.cuh"

#include "../../include/utils.h"

double transpose_distributed(float *out, float *in,
                             const int M, const int N,
                             const int M_split, const int N_split)
{
    float milliseconds = 0;

#ifdef DEBUG
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
#endif

    if (M_split <= 0 || N_split <= 0)
    {
        perror("M_split or N_split is not positive!");
        exit(EXIT_FAILURE);
    }

    // Calcular el tamaño de cada bloque
    int i, j;
    int i_size, j_size;
    int max_i_size = (M + M_split - 1) / M_split;
    int max_j_size = (N + N_split - 1) / N_split;

    // Reserva de memoria para la matriz de entrada y salida en la GPU
    float *d_in_sub, *d_out_sub;
    gpuErrchk(cudaMalloc((void **)&d_in_sub, max_i_size * max_j_size * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_out_sub, max_j_size * max_i_size * sizeof(float)));

#ifdef DEBUG
    gpuErrchk(cudaEventRecord(start));
#endif

    for (i = 0; i < M; i += max_i_size)
    {
        i_size = (i + max_i_size >= M) ? M - i : max_i_size;

        for (j = 0; j < N; j += max_j_size)
        {
            j_size = (j + max_j_size >= N) ? N - j : max_j_size;

            if (i_size * j_size == 1)
            {
                out[j * M + i] = in[i * N + j];
            }
            else
            {
                // Copiamos los datos necesarios
                gpuErrchk(cudaMemcpy2D((void *)d_in_sub, j_size * sizeof(float),
                                       (const void *)(in + i * N + j), N * sizeof(float),
                                       j_size * sizeof(float), i_size,
                                       cudaMemcpyHostToDevice));

                // Ajustar las dimensiones del grid y del bloque para el bloque actual
                dim3 blockDim(TILE_DIM, BLOCK_ROWS);
                dim3 gridDim((j_size + TILE_DIM - 1) / TILE_DIM,
                             (i_size + TILE_DIM - 1) / TILE_DIM);

                // Lanzamiento del kernel de transposición para el bloque actual
                cuda_transpose<<<gridDim, blockDim>>>(d_out_sub, d_in_sub, i_size, j_size);
                cudaCheckError();

                // Almacenamos el bloque de memoria en host, pero en la traspuesta
                // (traspuesta por bloques)
                gpuErrchk(cudaMemcpy2D((void *)(out + j * M + i), M * sizeof(float),
                                       (const void *)d_out_sub, i_size * sizeof(float),
                                       i_size * sizeof(float), j_size,
                                       cudaMemcpyDeviceToHost));
            }
        }
    }

#ifdef DEBUG
    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));
#else
    gpuErrchk(cudaDeviceSynchronize());
#endif

    // Libera la memoria de las matrices en la GPU y destruye los eventos
    gpuErrchk(cudaFree(d_in_sub));
    gpuErrchk(cudaFree(d_out_sub));

    return (double)milliseconds;
}