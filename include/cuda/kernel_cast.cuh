#ifndef __KERNEL_CAST_CUH__
#define __KERNEL_CAST_CUH__

#include <cuda_fp16.h>

/**
 * @brief Cast from float32 to float16 (half).
 *
 * @param A_ Half vector, N elems.
 * @param A Float vector, N elems.
 * @param N
 */
__global__ void f32_to_f16(half *A_, const float *A, const int N);

#endif