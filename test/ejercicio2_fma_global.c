#include <stdio.h>
#include <stdlib.h>

#include "../include/utils.h"

#include "../include/fma.h"
#include "../include/cuda/fma.cuh"
#include "../include/cuda/utils.cuh"

#define M 10
#define N 10
#define K 10

#define TOL (float)1e-4

int main()
{
    int errores;
    float A[M * K], B[K * N], C[M * N], D[M * N], D_GPU[M * N];
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

    // Ejecutamos la prueba de FMA en GPU (mem. global)
    exe_time_ms = fma_gpu_global(D, A, B, C, M, N, K);
    printf("GPU (global) took %fms\n", exe_time_ms);

    // Calculamos los errores entre ambas matrices
    errores = allequal(D, D_GPU, M, N, TOL);

    if (errores)
    {
        fprintf(stderr,
                "[AVISO] Existen %d diferencias entre el cómputo en CPU y en GPU (global) (tol. %f)\n",
                errores, TOL);

        if (M * N < 200)
        {
            print_mat(D, M, N);
            printf("\n");
            print_mat(D_GPU, M, N);
        }
    }

    return 0;
}