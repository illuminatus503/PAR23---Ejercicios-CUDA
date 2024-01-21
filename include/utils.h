#ifndef __CODECPU_H__
#define __CODECPU_H__

/**
 * @brief Genera matrices aleatorias para FMA.
 *
 * @param A float, M x K
 * @param B float, K x N
 * @param C float, M x N
 * @param M
 * @param N
 * @param K
 */
void gen_matrices(float *A, float *B, float *C, const int M, const int N, const int K);

/**
 * @brief Inicializa una matriz con valores float32 aleatorios.
 *
 * @param A float M x N
 * @param M
 * @param N
 */
void rand_init(float *A, const int M, const int N);

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

/**
 * @brief Muestra cómo quedaría la partición de una matriz,
 * usando unos parámetros de partición dados.
 *
 * @param A float, M x N
 * @param M
 * @param N
 * @param M_split Número de filas por las que dividir: M / M_split
 * @param N_split Número de columnas por las que dividir: N / N_split
 */
void print_split(float *A, const int M, const int N,
                 const int M_split, const int N_split);

/**
 * @brief Calcula si dos matrices son suficientemente similares, dada una
 * tolerancia.
 *
 * @param A float, M x N
 * @param B float, M x N
 * @param M
 * @param N
 * @param tol un valor float pequeño: por ejemplo, 1e-4
 * @return int 0 si son similares (dado tol.); > 0, si existen diferencias.
 */
int allequal(const float *A, const float *B,
             const int M, const int N,
             const float tol);

#endif