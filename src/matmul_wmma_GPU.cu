#include <cuda.h>
#include <mma.h>
#include <iostream>

#include "../include/utilities_CPU.h"
#include "../include/errchk_GPU.cuh"
#include "../include/matmul_wmma_GPU.cuh"

using namespace nvcuda;

__global__ void cuda_fma_wmma(float *A_, float *B_, float *C_, float *D,
                                   int N, int M, int P)
{
    // Define los fragmentos WMMA
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Inicializar el fragmento del acumulador
    wmma::fill_fragment(c_frag, 0.0f);

    // Cargar los fragmentos de las matrices A y B
    wmma::load_matrix_sync(a_frag, a, MATRIX_SIZE);
    wmma::load_matrix_sync(b_frag, b, MATRIX_SIZE);

    // Realizar la operación FMA
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Almacenar el resultado de vuelta en la matriz C
    wmma::store_matrix_sync(c, c_frag, MATRIX_SIZE, wmma::mem_row_major);
}

double __fma_wmma_GPU(float *A_, float *B_, float *C_, float *D,
                           int N, int M, int P)
{
    /**
     * Medición de tiempos
     */
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    /**
     * Variables de mem. device
     */
    const size_t size_A = N * M * sizeof(float);
    const size_t size_B = M * P * sizeof(float);
    const size_t size_C = N * P * sizeof(float);
    float *d_A, *d_B, *d_C, *d_D;

    gpuErrchk(cudaMalloc((void **)&d_A, size_A));
    gpuErrchk(cudaMalloc((void **)&d_B, size_B));
    gpuErrchk(cudaMalloc((void **)&d_C, size_C));
    gpuErrchk(cudaMalloc((void **)&d_D, size_C));

    // Copiamos los datos necesarios para las matrices A, B y C
    gpuErrchk(cudaMemcpy((void *)d_A, (const void *)A_, size_A, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_B, (const void *)B_, size_B, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C_, size_C, cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      threadsPerBlock: number of CUDA threads per grid block
    //      blocksPerGrid: number of blocks in grid
    dim3 threadsPerBlock(THR_PER_BLOCK, THR_PER_BLOCK);
    dim3 blocksPerGrid((P - 1) / threadsPerBlock.x + 1,
                       (N - 1) / threadsPerBlock.y + 1);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_fma_wmma<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N, M, P);
    cudaCheckError(); // Check error after execution
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array to host array
    gpuErrchk(cudaMemcpy((void *)D, (const void *)d_D, size_C, cudaMemcpyDeviceToHost));
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

double fma_wmma_GPU(float *A_, int N1, int M1,
                         float *B_, int N2, int M2,
                         float *C_, int N3, int M3,
                         float *D, int N, int M)
{
    if (!matrix_checkdims(N1, M1, N2, M2, N3, M3, N, M))
    {
        fprintf(stderr,
                "[DimError] La dimensiones de las matrices no coinciden: A(%d x %d) · B(%d x %d) + C(%d x %d) = D(%d x %d)\n",
                N1, M1, N2, M2, N3, M3, N, M);
        return 0.0; // Asum. que el checkeo no añade sobrecostes
    }

    return __fma_wmma_GPU(A_, B_, C_, D, N, M1, M);
}
