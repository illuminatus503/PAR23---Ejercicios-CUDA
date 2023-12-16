#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../include/codeCPU.h"

/**
 * @brief A partir de dos struct timestep, calcula el tiempo
 * de ejecución en ms.
 *
 * @param begin Medición al comienzo de la ejecución.
 * @param end Medición al final de la ejecución.
 * @return double Diferencia de tiempo entre mediciones, en ms.
 */
double timing_CPU(struct timespec begin, struct timespec end)
{
    return ((end.tv_sec - begin.tv_sec) * 1e3 + ((end.tv_nsec - begin.tv_nsec) * 1e-6));
}

void gen_matrices(int N, int M, int P, float *A, float *B, float *C)
{
    int i, j;

    // Inicializamos la matriz A
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            A[i * M + j] = rand() / ((float)RAND_MAX);
        }
    }

    // Inicializamos la matriz B
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < P; j++)
        {
            B[i * P + j] = rand() / ((float)RAND_MAX);
        }
    }

    // Inicializamos la matriz C
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < P; j++)
        {
            C[i * P + j] = rand() / ((float)RAND_MAX);
        }
    }
}

bool matrix_checkdims(int N1, int M1, int N2, int M2, int N3, int M3, int N, int M)
{
    if (M1 != N2) // Check: A(N1 x M1) · B(M1, M2) es posible
        return 0;

    if (N1 != N3) // Check: A(N1, ...) = C(N1, ...)
        return 0;

    if (M2 != M3) // Check: B(..., M2) = C(..., M2)
        return 0;

    if (N1 != N) // Check: A(N1, ...) = D(N1, ...)
        return 0;

    if (M2 != M) // Check: B(..., M2) = D(..., M2)
        return 0;

    return 1;
}

double __fmadd_CPU(float *A_, float *B_, float *C_, float *D, int N, int M, int P)
{
    int i, j, k;
    struct timespec begin, end;

    clock_gettime(CLOCK_MONOTONIC, &begin);
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < P; j++)
        {
            D[i * P + j] = C_[i * P + j];
            for (k = 0; k < M; k++)
            {
                D[i * P + j] += A_[i * M + k] * B_[k * P + j];
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    return timing_CPU(begin, end);
}

double fmadd_CPU(float *A_, int N1, int M1,
                 float *B_, int N2, int M2,
                 float *C_, int N3, int M3,
                 float *D, int N, int M)
{
    if (!matrix_checkdims(N1, M1, N2, M2, N3, M3, N, M))
    {
        fprintf(stderr,
                "[DimError] La dimensiones de las matrices no coinciden: A(%d x %d) · B(%d x %d) + C(%d x %d) = D(%d x %d)\n",
                N1, M1, N2, M2, N3, M3, N, M);
        return 0.0; // Asum. que el checkeo no añade sobrecostes
    }

    return __fmadd_CPU(A_, B_, C_, D, N, M1, M);
}

void matrix_print(float *A_, int N, int M)
{
    int i, j;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            printf("%3.3f ", A_[i * M + j]);
        }
        printf("\n");
    }
}

float matrix_infty_dist(float *A_, float *B_, int N, int M)
{
    int i, j;

    float row_sum;
    float infty_norm = 0.0;

    for (i = 0; i < N; i++)
    {
        row_sum = 0.0;
        for (j = 0; j < M; j++)
        {
            row_sum += (float)fabs(A_[i * M + j] - B_[i * M + j]);
        }

        if (row_sum > infty_norm)
        {
            infty_norm = row_sum;
        }
    }

    return infty_norm;
}