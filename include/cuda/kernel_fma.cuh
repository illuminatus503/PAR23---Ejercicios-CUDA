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

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void cuda_fma_wmma(half *A_, half *B_, float *C_, float *D,
                              int M_total, int N_total, int K_total,
                              int M_padded, int N_padded, int K_padded);

#endif