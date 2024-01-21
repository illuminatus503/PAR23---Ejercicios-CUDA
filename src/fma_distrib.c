#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>

#include "../include/utils.h"
#include "../include/fma.h"

double fma_cpu_distrib(float *D, float *A, float *B, float *C,
                       const int M, const int N, const int K,
                       const int M_split, const int N_split, const int K_split)
{
    int i, j, k;
    int i_sub, j_sub, k_sub;
    int i_size, j_size, k_size;

    float *A_sub, *B_sub, *C_sub, *D_sub;

    struct timespec begin, end;

    if (M_split <= 0 || N_split <= 0 || K_split <= 0)
    {
        perror("M_split, N_split or K_split is not positive!");
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

            // Inicializamos D_sub con elementos de C
            for (i_sub = 0; i_sub < i_size; i_sub++)
            {
                for (j_sub = 0; j_sub < j_size; j_sub++)
                {
                    D_sub[i_sub * N + j_sub] = C_sub[i_sub * N + j_sub];
                }
            }

            // Agregamos las multiplicaciones de matrices sucesivas
            for (k = 0; k < K; k += Ksub)
            {
                k_size = (k + Ksub > K) ? K - k : Ksub;

                // Apuntar a las submatrices correspondientes de A y B
                A_sub = (float *)&(A[i * K + k]);
                B_sub = (float *)&(B[k * N + j]);

                // Realizar la operación FMA en las submatrices

#pragma omp parallel for private(i_sub, j_sub, k_sub) schedule(static)
                for (i_sub = 0; i_sub < i_size; i_sub++)
                {
#pragma omp simd
                    for (j_sub = 0; j_sub < j_size; j_sub++)
                    {
                        for (k_sub = 0; k_sub < k_size; k_sub++)
                        {
                            D_sub[i_sub * N + j_sub] += A_sub[i_sub * K + k_sub] * B_sub[k_sub * N + j_sub];
                        }
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    return timing_cpu(begin, end);
}
