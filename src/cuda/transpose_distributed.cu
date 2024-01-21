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
    // Eventos para medir el tiempo
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    float milliseconds = 0;

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
    float *d_in_sub;
    gpuErrchk(cudaMalloc((void **)&d_in_sub, max_i_size * max_j_size * sizeof(float)));

    gpuErrchk(cudaEventRecord(start));

    for (i = 0; i < M; i += max_i_size)
    {
        i_size = (i + max_i_size >= M) ? M - i : max_i_size;

        for (j = 0; j < N; j += max_j_size)
        {
            j_size = (j + max_j_size >= N) ? N - j : max_j_size;

            // ! DEBUG print submatriz
            printf("Submatriz A[%d:%d, %d:%d]: \n", i, i + i_size, j, j + j_size);
            print_mat((const float *)(in + i * N + j), i_size, j_size);

            // Copiamos los datos necesarios
            gpuErrchk(cudaMemcpy2D((void *)d_in_sub, j_size * sizeof(float),          // dpitch: width of d_in_sub
                                   (const void *)(in + i * N + j), N * sizeof(float), // spitch: width of in
                                   j_size * sizeof(float), i_size,
                                   cudaMemcpyHostToDevice));

            // Ajustar las dimensiones del grid y del bloque para el bloque actual
            dim3 blockDim(TILE_DIM, BLOCK_ROWS);
            dim3 gridDim((i_size + TILE_DIM - 1) / TILE_DIM,  // Cambio: i_size en lugar de j_size
                         (j_size + TILE_DIM - 1) / TILE_DIM); // Cambio: j_size en lugar de i_size

            // Lanzamiento del kernel de transposición para el bloque actual
            cuda_transpose<<<gridDim, blockDim>>>(d_in_sub, d_in_sub, i_size, j_size);
            cudaCheckError();

            // Almacenamos el bloque de memoria en host
            gpuErrchk(cudaMemcpy2D((void *)(out + j * M + i), M * sizeof(float),   // dpitch: width of out
                                   (const void *)d_in_sub, i_size * sizeof(float), // spitch: width of d_in_sub
                                   i_size * sizeof(float), j_size,
                                   cudaMemcpyDeviceToHost));

            // ! DEBUG print submatriz
            printf("Submatriz At[%d:%d, %d:%d]: \n", j, j + j_size, i, i + i_size); // Cambio: j y i invertidos
            print_mat((const float *)(out + j * M + i), j_size, i_size);            // Cambio: j y i invertidos
            printf("......................................................................................\n");
        }
    }

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    // Libera la memoria de las matrices en la GPU y destruye los eventos
    gpuErrchk(cudaFree(d_in_sub));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    return (double)milliseconds;
}