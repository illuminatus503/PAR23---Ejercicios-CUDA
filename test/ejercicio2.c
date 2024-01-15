#include <stdio.h>
#include <stdlib.h>

#include "../include/utils.h"

#include "../include/fma.h"
#include "../include/cuda/fma.cuh"
#include "../include/cuda/utils.cuh"

/**
 * @brief Definimos las dimensiones de las matrices por defecto aquí.
 *
 */
#define N 101
#define M 227
#define P 990

int main()
{
    double time_naive_gpu, time_cpu;
    float *A, *B, *C, *D_cpu, *D_gpu;

    // Vamos a recoger información sobre las GPUs disponibles
    // antes de comenzar a operar.
    struct info_t gpu_array;
    load_gpu_info(&gpu_array);

    // Declaramos & init. las matrices en mem. din.
    A = (float *)malloc(N * M * sizeof(float));
    B = (float *)malloc(M * P * sizeof(float));
    C = (float *)malloc(N * P * sizeof(float));
    D_cpu = (float *)calloc(N * P, sizeof(float));
    D_gpu = (float *)calloc(N * P, sizeof(float));
    gen_matrices(N, M, P, A, B, C);

    // FMA en CPU
    time_cpu = fma_cpu(A, N, M, B, M, P, C, N, P, D_cpu, N, P);

    // FMA en GPU (memoria global)
    time_naive_gpu = fma_global_gpu(A, N, M, B, M, P, C, N, P, D_gpu, N, P, &gpu_array);

    printf("-----------------------------------------------------------\n");
    printf("Operation        | Exec. time (ms) | ||·||_inf / CPU vs GPU\n");
    printf("-----------------------------------------------------------\n");
    printf("FMA (CPU)        | %15.3f | %20.3f\n", time_cpu, 0.0);
    printf("FMA (GPU, naïve) | %15.3f | %20.3f\n", time_naive_gpu, matrix_infty_dist(D_cpu, D_gpu, N, P));
    printf("-----------------------------------------------------------\n");

    // Liberamos la memoria dinámica reservada
    clean_gpu_info(&gpu_array);
    free(A);
    free(B);
    free(C);
    free(D_cpu);
    free(D_gpu);

    return 0;
}
