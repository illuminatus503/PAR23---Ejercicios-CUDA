#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>

#include "../include/utils.h"
#include "../include/fma.h"

double fma_cpu(float *D, const float *A, const float *B, const float *C,
               const int M, const int N, const int K)
{
    int i, j, k;
    struct timespec begin, end;

    clock_gettime(CLOCK_MONOTONIC, &begin);
#pragma omp parallel for private(i, j, k) schedule(static)
    for (i = 0; i < M; i++)
    {
#pragma omp simd
        for (j = 0; j < N; j++)
        {
            D[i * N + j] = C[i * N + j];
            for (k = 0; k < K; k++)
            {
                D[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    return timing_cpu(begin, end);
}

void fma_cpu_nontiming(float *D, const float *A, const float *B, const float *C,
                       const int M, const int N, const int K)
{
    int i, j, k;

#pragma omp parallel for private(i, j, k) schedule(static)
    for (i = 0; i < M; i++)
    {
#pragma omp simd
        for (j = 0; j < N; j++)
        {
            D[i * N + j] = C[i * N + j];
            for (k = 0; k < K; k++)
            {
                D[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

double fma_cpu_distrib(float *D, const float *A, const float *B, const float *C,
                       const int M, const int N, const int K,
                       const int M_split, const int N_split, const int K_split)
{
    int i, j, k;
    int i_size, j_size, k_size;

    float *A_sub, *B_sub, *C_sub, *D_sub;

    struct timespec begin, end;

    if (M_split <= 0)
    {
        perror("M_split is not positive!");
        exit(EXIT_FAILURE);
    }

    if (N_split <= 0)
    {
        perror("N_split is not positive!");
        exit(EXIT_FAILURE);
    }

    if (K_split <= 0)
    {
        perror("K_split is not positive!");
        exit(EXIT_FAILURE);
    }

    // Calcular el tamaño de cada submatriz (considerando el padding si es necesario)
    int Msub = (M + M_split - 1) / M_split;
    int Nsub = (N + N_split - 1) / N_split;
    int Ksub = (K + K_split - 1) / K_split;

    // ! Operación FMA distribuída
    clock_gettime(CLOCK_MONOTONIC, &begin);

    for (i = 0; i < M; i += Msub)
    {
        i_size = (i + Msub > M) ? M - i : Msub;

        for (j = 0; j < N; j += Nsub)
        {
            j_size = (j + Nsub > N) ? N - j : Nsub;

            // Apuntar a la submatriz correspondiente de D y C
            D_sub = (float *)&(D[i * N + j]);
            C_sub = (float *)&(C[i * N + j]);

            // Agregamos las multiplicaciones de matrices sucesivas
            for (k = 0; k < K; k += Ksub)
            {
                k_size = (k + Ksub > K) ? K - k : Ksub;

                // Apuntar a las submatrices correspondientes de A y B
                A_sub = (float *)&(A[i * K + k]);
                B_sub = (float *)&(B[k * N + j]);

                // Realizar la operación FMA en las submatrices
                fma_cpu_nontiming(D_sub, A_sub, B_sub, C_sub, i_size, j_size, k_size);
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    return timing_cpu(begin, end);
}
