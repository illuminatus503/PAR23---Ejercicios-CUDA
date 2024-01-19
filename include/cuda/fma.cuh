#ifndef __FADD_CUH__
#define __FADD_CUH__

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
 * @param gpu_array un array de propiedades del sistema de aceleradores
 * actual
 * @return double Tiempo de ejecución de la operación en ms.
 */
double fma_global_gpu(float *A_, int N1, int M1,
                      float *B_, int N2, int M2,
                      float *C_, int N3, int M3,
                      float *D, int N, int M,
                      struct info_t *gpu_array);

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
 * @param gpu_array un array de propiedades del sistema de aceleradores
 * actual
 * @return double Tiempo de ejecución de la operación en ms.
 */
double fma_shared_gpu(float *A_, int N1, int M1,
                      float *B_, int N2, int M2,
                      float *C_, int N3, int M3,
                      float *D, int N, int M,
                      struct info_t *gpu_array);

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
 * @param gpu_array un array de propiedades del sistema de aceleradores
 * actual
 * @return double Tiempo de ejecución de la operación en ms.
 */
double fma_wmma_gpu(float *A_, int N1, int M1,
                    float *B_, int N2, int M2,
                    float *C_, int N3, int M3,
                    float *D, int N, int M);

#endif