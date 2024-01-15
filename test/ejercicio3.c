#include <stdio.h>
#include <stdlib.h>

#include "../include/utils.h"

#include "../include/fadd.h"
#include "../include/cuda/fadd.cuh"

/**
 * @brief Definimos las dimensiones de las matrices por defecto aquí.
 *
 */
#define N 101
#define M 227
#define P 990

int main()
{
    double time_shared_gpu, time_cpu;
    float *A, *B, *C, *D_cpu, *D_gpu;

    // Declaramos & init. las matrices en mem. din.
    A = (float *)malloc(N * M * sizeof(float));
    B = (float *)malloc(M * P * sizeof(float));
    C = (float *)malloc(N * P * sizeof(float));
    D_cpu = (float *)calloc(N * P, sizeof(float));
    D_gpu = (float *)calloc(N * P, sizeof(float));
    gen_matrices(N, M, P, A, B, C);

    printf("-----------------------------------------------------------\n");
    printf("Operation         | Exec. time (ms) | ||·||_inf / CPU vs GPU\n");
    printf("-----------------------------------------------------------\n");

    // FMA en CPU
    time_cpu = fma_CPU(A, N, M, B, M, P, C, N, P, D_cpu, N, P);
    printf("FMA (CPU)         | %15.3f | %20.3f\n", time_cpu, 0.0);

    // FMA en GPU (naïve)
    time_shared_gpu = fma_sharedmem_GPU(A, N, M, B, M, P, C, N, P, D_gpu, N, P);
    printf("FMA (GPU, shared) | %15.3f | %20.3f\n", time_shared_gpu, matrix_infty_dist(D_cpu, D_gpu, N, P));

    printf("-----------------------------------------------------------\n");

    // Liberamos la memoria dinámica reservada
    free(A);
    free(B);
    free(C);
    free(D_cpu);
    free(D_gpu);

    return 0;
}
