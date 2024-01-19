#include "../../include/cuda/kernel_fma.cuh"

__global__ void cuda_fma_global(float *D, const float *A_, const float *B_, const float *C_,
                                const int M, const int N, const int K)
{
    int i, j, k;
    float sum;

    i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M)
        return;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N)
        return;

    /**
     * Calcula el producto escalar de la fila i de A, columna j de B y
     * le suma el valor Cij: Dij = Cij + Ai_ · B_j
     */
    sum = C_[i * N + j];
    for (k = 0; k < K; k++)
    {
        sum += A_[i * K + k] * B_[k * N + j];
    }

    D[i * N + j] = sum;
}
