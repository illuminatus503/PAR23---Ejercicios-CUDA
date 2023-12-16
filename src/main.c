#include <stdio.h>
#include <stdlib.h>

#include "../include/codeCPU.h"
#include "../include/codeGPU.cuh"

int main(int argc, char *argv[])
{
    double time_gpu, time_cpu;
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

    // Declaramos & init. las matrices en mem. din.
    A = (float *)malloc(N * M * sizeof(float));
    B = (float *)malloc(M * P * sizeof(float));
    C = (float *)malloc(N * P * sizeof(float));
    D_cpu = (float *)calloc(N * P, sizeof(float));
    D_gpu = (float *)calloc(N * P, sizeof(float));
    gen_matrices(N, M, P, A, B, C);

    // Imprimimos el contenido de las matrices
    printf("A = \n");
    matrix_print(A, N, M);
    printf("\nB = \n");
    matrix_print(B, M, P);
    printf("\nC = \n");
    matrix_print(C, N, P);

    // Imprimimos el t. eje. en CPU de FMADD
    time_cpu = fmadd_CPU(A, N, M, B, M, P, C, N, P, D_cpu, N, P);
    printf("\nD_cpu (tej = %3.3f ms) = A · B + C \n", time_cpu);
    // matrix_print(D_cpu, N, P); // Ojo! Si son matrices muy grandes no se verán bien los resultados.

    // Imprimimos el t. eje. en GPU de FMADD
    time_gpu = fmadd_GPU(A, N, M, B, M, P, C, N, P, D_gpu, N, P);
    printf("\nD_gpu (tej = %3.3f ms) = A · B + C \n", time_gpu);
    // matrix_print(D_gpu, N, P); // Ojo! Si son matrices muy grandes no se verán bien los resultados.

    printf("\nMIDIENDO LAS DIFERENCIAS ENTRE CPU Y GPU: \n");
    printf("||D_cpu - D_gpu||_inf = %3.3f\n",
           matrix_infty_dist(D_cpu, D_gpu, N, P));

    // Liberamos la mem. din. reservada
    free(A);
    free(B);
    free(C);
    free(D_cpu);
    free(D_gpu);

    return 0;
}
