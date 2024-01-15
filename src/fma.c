#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/utils.h"
#include "../include/fma.h"

double __fma_cpu(float *A_, float *B_, float *C_, float *D, int N, int M, int P)
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

    return timing_cpu(begin, end);
}

double fma_cpu(float *A_, int N1, int M1,
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

    return __fma_cpu(A_, B_, C_, D, N, M1, M);
}