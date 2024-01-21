#ifndef __LINALG_CUH__
#define __LINALG_CUH__

#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Calcula la traspuesta de una matriz en CUDA.
 *
 * @param out float, N x M
 * @param in float M x N
 * @param M
 * @param N
 * @param M_split divisor de filas. Si M_split >= M, tomar filas de 1 e 1
 * @param N_split divisor de columnas. Si N_split >= N, tomar columnas de 1 e 1
 * @return double tiempo de ejecución en ms
 */
double transpose_cuda(float *out, const float *in, const int M, const int N,
                      const int M_split, const int N_split);

#endif