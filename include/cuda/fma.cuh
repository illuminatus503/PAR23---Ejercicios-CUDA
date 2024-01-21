#ifndef __FADD_CUH__
#define __FADD_CUH__

/**
 * @brief Fused-multiply add operation. En GPU, memoria global.
 *
 * @param D D = A * B + C, float M x N
 * @param A float M x K
 * @param B float K x N
 * @param C float M x N
 * @param M
 * @param N
 * @param K
 * @return double tiempo de ejecución en milisegundos
 */
double fma_gpu_global(float *D, float *A, float *B, float *C,
                      const int M, const int N, const int K);

/**
 * @brief Fused-multiply add operation. En GPU, usando memoria compartida.
 *
 * @param D D = A * B + C, float M x N
 * @param A float M x K
 * @param B float K x N
 * @param C float M x N
 * @param M
 * @param N
 * @param K
 * @return double tiempo de ejecución en milisegundos
 */
double fma_gpu_shared(float *D, float *A, float *B, float *C,
                      const int M, const int N, const int K);

/**
 * @brief Fused-multiply add operation. En GPU, usando tensor cores (WMMA).
 *
 * @param D D = A * B + C, float M x N
 * @param A float M x K
 * @param B float K x N
 * @param C float M x N
 * @param M
 * @param N
 * @param K
 * @return double tiempo de ejecución en milisegundos
 */
double fma_wmma_gpu(float *D, float *A, float *B, float *C,
                    const int M, const int N, const int K);

/**
 * @brief Distributed FMA (Fused-Multiply Add), using WMMA operations in CUDA 9.0+.
 * Compile using -arch=sm_75 or higher.
 *
 * @param D D = A * B + C. float, M x N
 * @param A float, M x K
 * @param B float, K x N
 * @param C float, M x N
 * @param M
 * @param N
 * @param K
 * @param M_split division of M dimension. If M_split >= M, process 1 row from D, per stream
 * @param N_split division of N dimension. If N_split >= N, process 1 col. from D, per stream
 * @param K_split division of K dimension. If K_split >= K, process 1 row/col. from A or B, per stream
 * @return double execution time, in miliseconds.
 */
double fma_wmma_gpu_distrib(float *D, float *A, float *B, float *C,
                            const int M, const int N, const int K,
                            const int M_split, const int N_split, const int K_split);

#endif