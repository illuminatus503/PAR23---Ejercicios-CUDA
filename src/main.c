#include <stdio.h>
#include <stdlib.h>

#include "../include/utils.h"
#include "../include/fadd.h"
#include "../include/cuda/fadd.cuh"
#include "../include/cuda/info.cuh"

int main(int argc, char *argv[])
{
    double time_naive_gpu, time_sharedmem_gpu, time_cpu;
    float *A, *B, *C, *D_cpu, *D_gpu;

    if (argc != 4)
    {
        fprintf(stderr,
                "Usage: %s [dim. N] [dim. M] [dim. P] \n",
                argv[0]);

        return 1;
    }

    // Recibimos las dimensiones de las matrices por stdin
    unsigned int N = strtoul(argv[1], NULL, 10);
    unsigned int M = strtoul(argv[2], NULL, 10);
    unsigned int P = strtoul(argv[3], NULL, 10);

    /**
     * @brief Ejecución del diagnóstico de las GPUs del sistema.
     *
     */
    print_gpu_info();

    // Declaramos & init. las matrices en mem. din.
    A = (float *)malloc(N * M * sizeof(float));
    B = (float *)malloc(M * P * sizeof(float));
    C = (float *)malloc(N * P * sizeof(float));
    D_cpu = (float *)calloc(N * P, sizeof(float));
    D_gpu = (float *)calloc(N * P, sizeof(float));
    gen_matrices(N, M, P, A, B, C);

    /**
     * @brief Ejecución de los tests sobre multiplicaciones
     *
     */
    printf("Operation, Exec. time (ms), ||·||_inf / CPU vs GPU\n");

    // ! FMA en CPU
    time_cpu = fma_CPU(A, N, M, B, M, P, C, N, P, D_cpu, N, P);
    printf("FMA (CPU), %3.3f, 0.0\n",
           time_cpu);

    // ! FMA en GPU (naïve)
    time_naive_gpu = fma_naive_GPU(A, N, M, B, M, P, C, N, P, D_gpu, N, P);
    printf("FMA (GPU, naïve), %3.3f, %3.3f\n",
           time_naive_gpu, matrix_infty_dist(D_cpu, D_gpu, N, P));

    // ! FMA en GPU (shared mem.)
    time_sharedmem_gpu = fma_sharedmem_GPU(A, N, M, B, M, P, C, N, P, D_gpu, N, P);
    printf("FMA (GPU, shared mem.), %3.3f, %3.3f\n",
           time_sharedmem_gpu, matrix_infty_dist(D_cpu, D_gpu, N, P));

    // Liberamos la mem. din. reservada
    free(A);
    free(B);
    free(C);
    free(D_cpu);
    free(D_gpu);

    return 0;
}
