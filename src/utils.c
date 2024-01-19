#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "../include/utils.h"

double timing_cpu(const struct timespec begin, const struct timespec end)
{
    return ((end.tv_sec - begin.tv_sec) * 1e3 + ((end.tv_nsec - begin.tv_nsec) * 1e-6));
}

void __init_rand(float *A, const int M, const int N)
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
    __init_rand(A, M, K);
    __init_rand(B, K, N);
    __init_rand(C, M, N);
}

bool matrix_checkdims(const int N1, const int M1,
                      const int N2, const int M2,
                      const int N3, const int M3,
                      const int N, const int M)
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

void print_mat(const float *A, const int M, const int N)
{
    int i, j;

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            printf("%3.3f ", A[i * N + j]);
        }
        printf("\n");
    }
}

int allequal(const float *A, const float *B, const int M, const int N, const float tol)
{
    int i;
    int errors = 0;

    float relative_err;

    for (i = 0; i < M * N; i++)
    {
        relative_err = fabs(A[i] - B[i]) / B[i];
        if (relative_err >= tol)
        {
            errors++;
        }
    }

    return errors;
}

float mse(const float *A, const float *B, const int M, const int N)
{
    int i;

    float diff;
    float error = 0.0;

    for (i = 0; i < M * N; i++)
    {
        diff = A[i] - B[i];
        error += diff * diff;
    }

    return error / (M * N);
}

void wmma_unpad(const float *A_padded, const int M_padded, const int N_padded,
                float *A, const int M, const int N)
{
    int i, j;

    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
        {
            A[i * N + j] = A_padded[i * N + j];
        }
    }
}

void wmma_pad(float *A, float *B, float *C,
              const int M, const int N, const int K,
              const int WMMA_M, const int WMMA_N, const int WMMA_K,
              float **A_padded, float **B_padded, float **C_padded,
              int *M_padded, int *N_padded, int *K_padded)
{
    int i, j;

    // Solamente se hace padding si es necesario. Es decir,
    // si alguna de las dimensiones no es múltiplo de 16.
    if (!(M % WMMA_M) && !(N % WMMA_N) && !(K % WMMA_K))
    {
        *A_padded = A;
        *B_padded = B;
        *C_padded = C;

        *M_padded = M;
        *N_padded = N;
        *K_padded = K;

        return;
    }

    // Dimensiones ajustadas (ceil)
    *M_padded = (M + WMMA_M - 1) / WMMA_M * WMMA_M;
    *N_padded = (N + WMMA_N - 1) / WMMA_N * WMMA_N;
    *K_padded = (K + WMMA_K - 1) / WMMA_K * WMMA_K;

    // Reservamos memoria
    *A_padded = (float *)calloc((*M_padded) * (*K_padded), sizeof(float));
    *B_padded = (float *)calloc((*K_padded) * (*N_padded), sizeof(float));
    *C_padded = (float *)calloc((*M_padded) * (*N_padded), sizeof(float));

    // Copiamos los datos en las posiciones adecuadas
    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < K; ++j)
        {
            (*A_padded)[i * K + j] = A[i * K + j];
        }
    }

    for (i = 0; i < K; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            (*B_padded)[i * N + j] = B[i * N + j];
        }
    }

    for (i = 0; i < M; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            (*C_padded)[i * N + j] = C[i * N + j];
        }
    }
}
