#ifndef __CODECPU_H__
#define __CODECPU_H__

#include <stdlib.h>

/**
 * @brief Genera tres matrices compatibles para la multiplicación, u
 * operaciones FMADD (Fused-Multiply-Add).
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
 * @brief Operación FMADD: Fused-Multiply-Add
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
double fmadd_CPU(float *A_, int N1, int M1,
                  float *B_, int N2, int M2,
                  float *C_, int N3, int M3,
                  float *D, int N, int M);

/**
 * @brief Checkea si las dimensiones de las matrices son compatibles
 * con las operaciones FMADD.
 *
 * @param N1
 * @param M1
 * @param N2
 * @param M2
 * @param N3
 * @param M3
 * @param N
 * @param M
 * @return true Si las dimensiones son compatibles con FMADD
 * @return false En cualquier otro caso
 */
bool matrix_checkdims(int N1, int M1,
                      int N2, int M2,
                      int N3, int M3,
                      int N, int M);

/**
 * @brief Imprime una matriz por pantalla, siempre que esté en memoria.
 *
 * @param A_
 * @param N
 * @param M
 */
void matrix_print(float *A_, int N, int M);

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

#endif