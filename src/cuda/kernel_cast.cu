#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "../../include/cuda/kernel_cast.cuh"

__global__ void f32_to_f16(half *A_, const float *A, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        A_[i] = A[i];
    }
}
