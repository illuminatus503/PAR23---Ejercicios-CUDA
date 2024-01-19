#ifndef __CODECPU_H__
#define __CODECPU_H__

/**
 * @brief Genera tres matrices compatibles para la multiplicación, u
 * operaciones FMA (Fused-Multiply-Add).
 *
 *
 * @param A float, M x K
 * @param B float, K x N
 * @param C float, M x N
 * @param M
 * @param N
 * @param K
 */
void gen_matrices(float *A, float *B, float *C,
                  const int M, const int N, const int K);

/**
 * @brief Checkea si las dimensiones de las matrices son compatibles
 * con las operaciones FMA.
 *
 * @param N1
 * @param M1
 * @param N2
 * @param M2
 * @param N3
 * @param M3
 * @param N
 * @param M
 * @return true Si las dimensiones son compatibles con FMA
 * @return false En cualquier otro caso
 */
bool matrix_checkdims(const int N1, const int M1,
                      const int N2, const int M2,
                      const int N3, const int M3,
                      const int N, const int M);

/**
 * @brief Imprime una matriz por pantalla, siempre que esté en memoria.
 *
 * @param A float, M x N
 * @param N
 * @param M
 */
void print_mat(const float *A, const int M, const int N);

/**
 * @brief Timing in CPU.
 *
 * @param begin Begin measurement
 * @param end End measurement
 * @return double Elapsed time between measurements, in ms.
 */
double timing_cpu(const struct timespec begin,
                  const struct timespec end);

/**
 * @brief Retorna el número de diferencias entre ambas matrices.
 * Sin ambas matrices son iguales, según una toleracia, retorna 0.
 *
 * @param A float, M x N
 * @param B float, M x N
 * @param M
 * @param N
 * @param tol un eps pequeño, como 1e-4
 * @return int 0 si no hay errores significativos; > 0 si no.
 */
int allequal(const float *A, const float *B, const int M, const int N,
             const float tol);

/**
 * @brief Mean Squeared Error (MSE). Calculamos el error cuadrático
 * promedio de la diferencia de las matrices.
 *
 * @param A float, M x N
 * @param B float, M x N
 * @param M
 * @param N
 * @return float el MSE
 */
float mse(const float *A, const float *B, const int M, const int N);

/**
 * @brief Rellena las matrices A, B y C de las operación
 *      C = alpha * (A * B) + beta * C
 *
 *  con A(M x K), B(K x N) y C(M x N).
 *
 * @param A
 * @param B
 * @param C
 * @param M
 * @param N
 * @param K
 * @param WMMA_M Múltiplo al que rellenar las filas
 * @param WMMA_N Múltiplo al que rellenar las columnas
 * @param WMMA_K Múltiplo al que rellenar la dim. oculta
 * @param A_padded Matriz rellena por el final
 * @param B_padded Matriz rellena por el final
 * @param C_padded Matriz rellena por el final
 * @param M_padded Dimensión de filas ajustada
 * @param N_padded Dimensión de columnas ajustada
 * @param K_padded Dimensión de oculta ajustada
 */
void wmma_pad(float *A, float *B, float *C,
              const int M, const int N, const int K,
              const int WMMA_M, const int WMMA_N, const int WMMA_K,
              float **A_padded, float **B_padded, float **C_padded,
              int *M_padded, int *N_padded, int *K_padded);

/**
 * @brief Remove padding from a padded matrix.
 *
 * @param A_padded
 * @param M_padded
 * @param N_padded
 * @param A
 * @param M
 * @param N
 */
void wmma_unpad(const float *A_padded, const int M_padded, const int N_padded,
                float *A, const int M, const int N);

#endif