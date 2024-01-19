#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../include/utils.h"

double timing_cpu(struct timespec begin, struct timespec end)
{
    return ((end.tv_sec - begin.tv_sec) * 1e3 + ((end.tv_nsec - begin.tv_nsec) * 1e-6));
}

void __rand_init(float *A, const int M, const int N)
{
    int i, j;

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = rand() / ((float)RAND_MAX);
        }
    }
}

void gen_matrices(float *A, float *B, float *C,
                  const int M, const int N, const int K)
{
    __rand_init(A, M, K);
    __rand_init(B, K, N);
    __rand_init(C, M, N);
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

void print_mat(float *A_, int N, int M)
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

float mse(float *A_, float *B_, int rows, int cols)
{
    float error = 0.0;
    int size = rows * cols;

    for (int i = 0; i < size; i++)
    {
        float diff = A_[i] - B_[i];
        error += diff * diff;
    }

    return error / size;
}