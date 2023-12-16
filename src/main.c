#include <stdio.h>
#include <stdlib.h>

#include "../include/codeCPU.h"
#include "../include/codeGPU.cuh"

int main(int argc, char *argv[])
{
    float *A, *B, *C;
    float *C_local;
    double time_gpu, time_cpu;

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s length\n", argv[0]);
        return (0);
    }

    unsigned int N = strtoul(argv[1], NULL, 10);

    A = (float *)malloc(N * sizeof(float));
    B = (float *)malloc(N * sizeof(float));
    C = (float *)malloc(N * sizeof(float));
    C_local = (float *)malloc(N * sizeof(float));

    init_vectors(A, B, N);
    time_cpu = add_vectors_CPU(A, B, C, N);
    time_gpu = add_vectors_GPU(A, B, C, N);

    printf("TIME CPU: %.3fs; TIME GPU: %.3fs\n",
           time_cpu, time_gpu);

    free(A);
    free(B);
    free(C);
    free(C_local);

    return (0);
}
