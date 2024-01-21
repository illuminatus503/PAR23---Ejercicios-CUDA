#ifndef __KERNEL_ADD_CUH__
#define __KERNEL_ADD_CUH__

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Add two (flatten) matrices inplace. B += A.
 *
 * @param B float, M x N
 * @param A float, M x N
 * @param N number of elements in A or B
 */
__global__ void cuda_add_inplace(float *B, const float *A, const int N);

#endif