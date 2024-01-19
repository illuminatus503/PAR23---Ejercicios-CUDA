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
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            D[i * N + j] = C[i * N + j];
            for (k = 0; k < K; k++)
            {
                D[i * N + j] += A[i * K + k] * B[k * K + j];
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    return timing_cpu(begin, end);
}
