#include "../../include/cuda/kernel_fma.cuh"

__global__ void cuda_gemm_global(float *C, const float *A, const float *B,
                                 const int M, const int N, const int K,
                                 const float alpha, const float beta)
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
    sum = 0.0f;
    for (k = 0; k < K; k++)
    {
        sum += A[i * K + k] * B[k * N + j];
    }

    C[i * N + j] = alpha * sum + beta * C[i * N + j];
}
