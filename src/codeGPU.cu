#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
// #include <cuda_runtime.h>

#include "../include/codeCPU.h"
#include "../include/codeGPU.cuh"

#define THR_PER_BLOCK 16

__global__ void cuda_matmul(float *A_, float *B_, float *C_, float *D,
                            int N, int M, int P)
{
    int i, j, k;
    float sum;

    /**
     * Calculamos el índice de i (filas, dim. y)
     */
    i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= N)
        return;

    /**
     * Calculamos el índice de j (columnas, dim. x)
     */
    j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= P)
        return;

    /**
     * Calcula el producto escalar de la fila i de A, columna j de B y
     * le suma el valor Cij: el resultado se guarda en Dij.
     *                      Dij = Cij + Ai_ · B_j
     */
    sum = C_[i * P + j];
    for (k = 0; k < M; k++)
    {
        sum += A_[i * M + k] * B_[k * P + j];
    }

    D[i * P + j] = sum; // solo escribimos una vez en mem. global de device
}

double __fmadd_GPU(float *A_, float *B_, float *C_, float *D,
                   int N, int M, int P)
{
    /**
     * Medición de tiempos
     */
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /**
     * Variables de mem. device
     */
    const unsigned int size_A = N * M * sizeof(float);
    const unsigned int size_B = M * P * sizeof(float);
    const unsigned int size_C = N * P * sizeof(float);
    float *d_A, *d_B, *d_C, *d_D;

    gpuErrchk(cudaMalloc((void **)&d_A, size_A));
    gpuErrchk(cudaMalloc((void **)&d_B, size_B));
    gpuErrchk(cudaMalloc((void **)&d_C, size_C));
    gpuErrchk(cudaMalloc((void **)&d_D, size_C));

    // Copiamos los datos necesarios para las matrices A, B y C
    gpuErrchk(cudaMemcpy(d_A, A_, size_A, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B_, size_B, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_C, C_, size_C, cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      threadsPerBlock: number of CUDA threads per grid block
    //      blocksPerGrid: number of blocks in grid
    dim3 threadsPerBlock(THR_PER_BLOCK, THR_PER_BLOCK);
    dim3 blocksPerGrid((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N, M, P);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array to host array
    cudaMemcpy(D, d_D, size_C, cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&exe_time_ms, start, stop);

    /**
     * Free CUDA mem.
     */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);

    return (double)exe_time_ms;
}

double fmadd_GPU(float *A_, int N1, int M1,
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

    return __fmadd_GPU(A_, B_, C_, D, N, M1, M);
}
