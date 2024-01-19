#ifndef __FADD_CUH__
#define __FADD_CUH__

/**
 * @brief Operación FMA (fused multiply-add) en GPU, memoria global.
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
double fma_gpu_global(float *D, const float *A, const float *B, const float *C,
                      const int M, const int N, const int K);

/**
 * @brief Operación FMA (fused multiply-add) en GPU, usando memoria compartida
 * de bloque.
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
double fma_gpu_shared(float *D, const float *A, const float *B, const float *C,
                      const int M, const int N, const int K);

/**
 * @brief Operación FMA (fused multiply-add) en GPU, usando tensor cores.
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
double fma_gpu_wmma(float *D, const float *A, const float *B, const float *C,
                    const int M, const int N, const int K);

#endif