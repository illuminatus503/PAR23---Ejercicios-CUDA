#include "../../include/cuda/kernel_add.cuh"

__global__ void cuda_add_inplace(float *B, const float *A, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        B[i] += A[i];
    }
}