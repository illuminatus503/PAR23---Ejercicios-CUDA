#include <stdio.h>
#include <stdlib.h>

#include "../include/utils.h"

#include "../include/fma.h"
#include "../include/cuda/fma.cuh"
#include "../include/cuda/utils.cuh"

#define M 100
#define N 1000
#define K 100

int main()
{
    float A[M * K], B[K * N], C[M * N], D[M * N];
    double exe_time_ms;

    // ! TEST: las dimensiones de las matrices son compatibles?
    if (!matrix_checkdims(M, K, K, N, M, N, M, N))
    {
        fprintf(stderr,
                "[DimError] La dimensiones de las matrices no coinciden: A(%d x %d) · B(%d x %d) + C(%d x %d) = D(%d x %d)\n",
                M, K, K, N, M, N, M, N);
        return 1;
    }

    // Generamos las matrices
    gen_matrices(A, B, C, M, N, K);

    // Ejecutamos la prueba de FMA
    exe_time_ms = fma_cpu(D, A, B, C, M, N, K);
    printf("CPU took %fms\n", exe_time_ms);

    // Imprimimos por pantalla el resultado
    // print_mat(D, M, N);

    return 0;
}