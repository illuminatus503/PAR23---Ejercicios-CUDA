#ifndef __MATMUL_CPU__
#define __MATMUL_CPU__

/**
 * @brief Operación FMA (fused multiply-add) en CPU.
 *
 * @param D Matriz de salida: D = A * B + C, float M x N
 * @param A float, M x K
 * @param B float, K x N
 * @param C float, M x N
 * @param N
 * @param M
 * @param K
 * @return double Tiempo de ejecución de la operación, en milisegundos
 */
double fma_cpu(float *D, const float *A, const float *B, const float *C,
               const int M, const int N, const int K);

#endif