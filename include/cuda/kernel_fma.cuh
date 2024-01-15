#ifndef __KERNEL_FADD_CUH__
#define __KERNEL_FADD_CUH__

#define TILE_SIZE 32 // Tile de T x T hilos, por bloque: se recomienda que TILE_SIZE == THR_PER_BLOCK

__global__ void cuda_fma_global(float *A_, float *B_, float *C_, float *D,
                                int N, int M, int P);

__global__ void cuda_fma_sharedmem(float *A_, float *B_, float *C_, float *D,
                                   int N, int M, int P);

#endif