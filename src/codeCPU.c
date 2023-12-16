#include <stdio.h>
#include <time.h>

#include "../include/codeCPU.h"

double timing_CPU(struct timespec begin, struct timespec end)
{
    return ((end.tv_sec - begin.tv_sec) + ((end.tv_nsec - begin.tv_nsec) / 1000000000.0));
}

void init_vectors(float *A, float *B, int N)
{
    int i;
    for (i = 0; i < N; i++)
    {
        A[i] = 1.0;
        B[i] = 1.0;
    }
}

double add_vectors_CPU(float *A, float *B, float *C, int N)
{
    int i;
    struct timespec begin, end;

    clock_gettime(CLOCK_MONOTONIC, &begin);

    for (i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    return (timing_CPU(begin, end));
}

double prod_vectors_CPU(float *A, float *B, float *C, int N)
{
    int i;
    struct timespec begin, end;

    clock_gettime(CLOCK_MONOTONIC, &begin);

    for (i = 0; i < N; i++)
    {
        C[i] = A[i] * B[i];
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    return (timing_CPU(begin, end));
}

double dot_prod_vectors_CPU(float *A, float *B, float *C, int N)
{
    int i;
    float tmp = 0;
    struct timespec begin, end;

    clock_gettime(CLOCK_MONOTONIC, &begin);

    for (i = 0; i < N; i++)
    {
        tmp += A[i] * B[i];
    }
    *C = tmp;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (timing_CPU(begin, end));
}

void print_vector(float *C, int N)
{
    int i;

    for (i = 0; i < N; i++)
    {
        printf("%f ", C[i]);
    }
    printf("\n");
}
