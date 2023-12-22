#ifndef __MATMUL_SHAREDMEM_GPU_CUH__
#define __MATMUL_SHAREDMEM_GPU_CUH__

#define TILE_SIZE 32     // Tile de T x T hilos, por bloque: se recomienda que TILE_SIZE == THR_PER_BLOCK
#define THR_PER_BLOCK 32 // Número de hilos por bloque

/**
 * @brief Operación FMADD: Fused-Multiply-Add en GPU.
 * Utilizando memoria compartida.
 *
 * @param A_ Matriz A(N x M1) de float de entrada
 * @param N1
 * @param M1
 * @param B_ Matriz B(M1 x M) de float de entrada
 * @param N2
 * @param M2
 * @param C_ Matriz C(N x M) de float de entrada
 * @param N3
 * @param M3
 * @param D Matriz D(N x M) de float
 * @param N
 * @param M
 * @return double Tiempo de ejecución de la operación en ms.
 */
double fma_sharedmem_GPU(float *A_, int N1, int M1,
                         float *B_, int N2, int M2,
                         float *C_, int N3, int M3,
                         float *D, int N, int M);

#endif
