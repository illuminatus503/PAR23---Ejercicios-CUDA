#ifndef __KERNEL_FADD_CUH__
#define __KERNEL_FADD_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // soporte para half en cuda

#define WARP_SIZE 32
#define TILE_SIZE 16

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void cuda_gemm_global(float *C, const float *A, const float *B,
                                 const int M, const int N, const int K,
                                 const float alpha, const float beta);

__global__ void cuda_gemm_shared(float *C, const float *A, const float *B,
                                 const int M, const int N, const int K,
                                 const float alpha, const float beta);

__global__ void cuda_gemm_wmma(float *C, const half *A, const half *B,
                               const int M, const int N, const int K,
                               const float alpha, const float beta);

/**
 * @brief Convert from F32 to F16, on GPU.
 *
 * @param out Output F16 flatten mat.
 * @param in Input F32 flatten mat.
 * @param n Total number of elems. of the flatten mat.
 */
__global__ void convertFp32ToFp16(half *out, float *in, int n);

#endif