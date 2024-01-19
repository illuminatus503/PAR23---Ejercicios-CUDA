#ifndef __CODECPU_H__
#define __CODECPU_H__

/**
 * @brief Genera tres matrices compatibles para la multiplicación, u
 * operaciones FMA (Fused-Multiply-Add).
 *
 * @param N Dimensión N
 * @param M Dimensión M
 * @param P Dimensión P
 * @param A Matriz A(N x M) de float
 * @param B Matriz B(M x P) de float
 * @param C Matriz C(N x P) de float
 */
void gen_matrices(int N, int M, int P, float *A, float *B, float *C);

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
bool matrix_checkdims(int N1, int M1,
                      int N2, int M2,
                      int N3, int M3,
                      int N, int M);

/**
 * @brief Calcula la distancia infinito entre dos
 * matrices, inducida por la norma infinito de una
 * matriz:
 *
 *      ||A - B||_infty = max_{i=0..nfilas-1} suma_{j=0..ncolumnas-1}{abs(Aij - Bij)}
 *
 * Para comparar las matrices generadas por los métodos
 * en serie y vectorial.
 *
 * @param A_ Matriz de float A(N x M)
 * @param B_ Matriz de float B(N x M)
 * @param N
 * @param M
 * @return float La norma infinito de la diferencia de A y B.
 */
float matrix_infty_dist(float *A_, float *B_, int N, int M);

/**
 * @brief Imprime una matriz por pantalla, siempre que esté en memoria.
 *
 * @param A_
 * @param N
 * @param M
 */
void print_mat(float *A_, int N, int M);

/**
 * @brief Timing in CPU.
 *
 * @param begin Begin measurement
 * @param end End measurement
 * @return double Elapsed time between measurements, in ms.
 */
double timing_cpu(struct timespec begin, struct timespec end);

/**
 * @brief Mean Squeared Error (MSE). Calculamos el error cuadrático
 * promedio de la diferencia de las matrices.
 *
 * @param A_ matriz de float
 * @param B_ matriz de float
 * @param rows filas
 * @param cols columnas
 * @return float El MSE entre ambas
 */
float mse(float *A_, float *B_, int rows, int cols);

#endif