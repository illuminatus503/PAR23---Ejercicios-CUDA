#include <stdio.h>
#include <stdlib.h>

#include "../include/utils.h"

#include "../include/fma.h"
#include "../include/cuda/fma.cuh"
#include "../include/cuda/utils.cuh"

#define N 10
#define M 5
#define P 1

int main()
{
    // Generamos matrices para la operación
    float *A = (float *)malloc(N * M * sizeof(float));
    float *B = (float *)malloc(M * P * sizeof(float));
    float *C = (float *)malloc(N * P * sizeof(float));
    float *D_cpu = (float *)malloc(N * P * sizeof(float));
    float *D_gpu = (float *)malloc(N * P * sizeof(float));
    gen_matrices(N, M, P, A, B, C);

    // Calculamos la operación FMA en CPU & imprimimos
    fma_cpu(A, N, M,
            B, M, P,
            C, N, P,
            D_cpu, N, P);
    printf(" ** FMA in CPU **\n");
    print_mat(D_cpu, N, P);

    // Calculamos la misma operación en GPU & imprimimos
    struct info_t gpu_array;
    load_gpu_info(&gpu_array);

    fma_wmma_gpu(A, N, M,
                 B, M, P,
                 C, N, P,
                 D_gpu, N, P,
                 &gpu_array);

    clean_gpu_info(&gpu_array);
    printf(" ** FMA in GPU (MMA) **\n");
    print_mat(D_gpu, N, P);

    // Free memoria dinámica.
    free(A);
    free(B);
    free(C);
    free(D_cpu);
    free(D_gpu);

    return 0;
}