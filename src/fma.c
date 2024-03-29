#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <omp.h>

#include "../include/utils.h"
#include "../include/fma.h"

double fma_cpu(float *D, float *A, float *B, float *C,
               const int M, const int N, const int K)
{
    int i, j, k;

#ifdef DEBUG
    struct timespec begin, end;
    clock_gettime(CLOCK_MONOTONIC, &begin);
#endif

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

#ifdef DEBUG
    clock_gettime(CLOCK_MONOTONIC, &end);
    return timing_cpu(begin, end);
#else
    return 0.0;
#endif
}
