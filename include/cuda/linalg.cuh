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
 * @return double tiempo de ejecución en ms
 */
double transpose_cuda(float *out, float *in, const int M, const int N);

/**
 * @brief Calcula la traspuesta de una matriz, en CUDA distribuído.
 * 
 * @param out float, N x M
 * @param in float M x N
 * @param M 
 * @param N 
 * @param M_split 
 * @param N_split 
 * @return double tiempo de ejecución en ms
 */
double transpose_distributed(float *out, float *in, const int M, const int N, const int M_split, const int N_split);

#endif