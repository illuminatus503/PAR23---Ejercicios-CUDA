#include "../../include/cuda/kernel_fma.cuh"

__global__ void cuda_fma_global(float *A_, float *B_, float *C_, float *D,
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
