#ifndef __CODE_GPU_CUH__
#define __CODE_GPU_CUH__

// #include <cuda.h>
// #include <cuda_runtime_api.h>

#include <stdbool.h>

/* Macro for checking cuda errors following a cuda launch or api call
 Taken from: https://gist.github.com/jefflarkin/5390993 */
#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess)                                                              \
    {                                                                                  \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(0);                                                                         \
    }                                                                                  \
  }

#define gpuErrchk(call)                                       \
  do                                                          \
  {                                                           \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess)                                   \
    {                                                         \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

/**
 * @brief Operación FMADD: Fused-Multiply-Add en GPU
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
double fmadd_GPU(float *A_, int N1, int M1,
                 float *B_, int N2, int M2,
                 float *C_, int N3, int M3,
                 float *D, int N, int M);

#endif
