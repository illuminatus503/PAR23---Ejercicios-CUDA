#ifndef __FADD_CUH__
#define __FADD_CUH__

/**
 * @brief Fused-multiply add operation. En GPU, memoria global.
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
double fma_gpu_global(float *D, const float *A, const float *B, const float *C,
                      const int M, const int N, const int K);

/**
 * @brief Fused-multiply add operation. En GPU, usando memoria compartida.
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
double fma_gpu_shared(float *D, const float *A, const float *B, const float *C,
                      const int M, const int N, const int K);

/**
 * @brief Fused-multiply add operation. En GPU, usando tensor cores (WMMA).
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
double fma_wmma_gpu(float *D, const float *A, const float *B, const float *C,
                    const int M, const int N, const int K);

#endif