#ifndef __KERNEL_FADD_CUH__
#define __KERNEL_FADD_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // soporte para half en cuda

#define THR_PER_BLOCK 32 // Número de hilos por bloque, por dimensión X, Y

__global__ void cuda_fma_global(float *A_, float *B_, float *C_, float *D,
                                int N, int M, int P);

#define TILE_SIZE 32 // Tile de T x T hilos, por bloque: se recomienda que TILE_SIZE == THR_PER_BLOCK

__global__ void cuda_fma_shared(float *A_, float *B_, float *C_, float *D,
                                int N, int M, int P);

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void cuda_fma_wmma(half *a, half *b, float *c, int M, int N, int K, float alpha, float beta);

#endif