#ifndef __MATMUL_CPU__
#define __MATMUL_CPU__

/**
 * @brief Fused-multiply add operation. En CPU.
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
double fma_cpu(float *D, const float *A, const float *B, const float *C,
               const int M, const int N, const int K);

/**
 * @brief Fused-multiply add operation. En CPU, procesamiento por bloques.
 * 
 * Prueba para comprobar el modo en el que se dividen las multiplicaciones
 * de matrices entre diferentes procesos.
 *
 * @param D D = A * B + C, float M x N
 * @param A float M x K
 * @param B float K x N
 * @param C float M x N
 * @param M
 * @param N
 * @param K
 * @param M_split División de M (filas de D)
 * @param N_split División de N (columnas de D)
 * @param K_split División de K (filas/columnas de las matrices A, B, resp.)
 * @return double tiempo de ejecución en milisegundos
 */
double fma_cpu_distrib(float *D, const float *A, const float *B, const float *C,
                       const int M, const int N, const int K,
                       const int M_split, const int N_split, const int K_split);

#endif