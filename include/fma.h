#ifndef __MATMUL_CPU__
#define __MATMUL_CPU__

/**
 * @brief Fused-multiply add operation. En CPU.
 *
 * @param D D = A * B + C, float M x N
 * @param A float M x N
 * @param B float N x K
 * @param C float M x N
 * @param M
 * @param N
 * @param K
 * @return double tiempo de ejecución en milisegundos
 */
double fma_cpu(float *D, const float *A, const float *B, const float *C,
               const int M, const int N, const int K);

#endif