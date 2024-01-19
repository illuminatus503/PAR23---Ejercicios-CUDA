#ifndef __KERNEL_FADD_CUH__
#define __KERNEL_FADD_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h> // soporte para half en cuda

#define WARP_SIZE 32 // Número de hilos por bloque, por dimensión X, Y
#define TILE_SIZE 32 // Tile de T x T hilos, por bloque: se recomienda que TILE_SIZE == WARP_SIZE

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void cuda_fma_global(float *C, const float *A, const float *B,
                                const int M, const int N, const int K,
                                const float alpha, const float beta);

__global__ void cuda_fma_shared(float *C, const float *A, const float *B,
                                const int M, const int N, const int K,
                                const float alpha, const float beta);

__global__ void cuda_fma_wmma(float *C, const half *A, const half *B,
                              const int M, const int N, const int K,
                              const float alpha, const float beta);

#endif