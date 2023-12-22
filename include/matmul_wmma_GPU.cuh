#ifndef __MATMUL_WMMA_GPU_CUH__
#define __MATMUL_WMMA_GPU_CUH__

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

/**
 * @brief Operación FMADD: Fused-Multiply-Add en GPU.
 * Utilizando tensor cores & a API Warp Matrix Multiply 
 * Accumulate (WMMA).
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
double fma_wmma_GPU(float *A_, int N1, int M1,
                    float *B_, int N2, int M2,
                    float *C_, int N3, int M3,
                    float *D, int N, int M);

#endif
