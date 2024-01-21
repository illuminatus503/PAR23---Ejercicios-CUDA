#include <cuda.h>
#include <cuda_runtime.h>

#include "../../include/cuda/error.cuh"
#include "../../include/cuda/kernel_linalg.cuh"
#include "../../include/cuda/linalg.cuh"

double transpose_cuda(float *out, const float *in, const int M, const int N,
                      const int M_split, const int N_split)
{
    // Eventos para medir el tiempo
    cudaEvent_t start, stop;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));
    float milliseconds = 0;

    // Reserva de memoria para la matriz de entrada y salida en la GPU
    float *d_in, *d_out;
    gpuErrchk(cudaMalloc((void **)&d_in, M * N * sizeof(float)));
    gpuErrchk(cudaMalloc((void **)&d_out, N * M * sizeof(float)));

    // Copiar la matriz de entrada al dispositivo
    gpuErrchk(cudaMemcpy(d_in, in, M * N * sizeof(float), cudaMemcpyHostToDevice));

    gpuErrchk(cudaEventRecord(start));

    // Calcular dimensiones del grid y bloque para toda la matriz
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Lanzamiento del kernel de transposición para toda la matriz
    cuda_transpose<<<gridDim, blockDim>>>(d_out, d_in, N, M); // Cambiar M por N y viceversa
    cudaCheckError();
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaEventRecord(stop));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copiar la matriz transpuesta de vuelta al host
    gpuErrchk(cudaMemcpy(out, d_out, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    // Libera la memoria de las matrices en la GPU y destruye los eventos
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
    gpuErrchk(cudaEventDestroy(start));
    gpuErrchk(cudaEventDestroy(stop));

    return (double)milliseconds;
}