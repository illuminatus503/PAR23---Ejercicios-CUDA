#include <stdio.h>
#include <stdlib.h>

#include "../include/utils.h"

#include "../include/fma.h"
#include "../include/cuda/fma.cuh"
#include "../include/cuda/utils.cuh"

#define M 10
#define N 9
#define K 4

#define M_split 1 // si M_split => M, entonces, se toman filas de 1 en 1
#define N_split 1 // si N_split => N, entonces, se toman columnas de 1 en 1
#define K_split 1 // si K_split >= K, entonces, se toman filas/columnas de 1 en 1

#define TOL (float)1e-4

int main()
{
    float A[M * K], B[K * N], C[M * N], D[M * N], D_DISTRIB[M * N];
    double exe_time_ms;
    int errores;

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

    printf("[+] Ejecutando prueba ESTÁNDAR\n");
    exe_time_ms = fma_cpu(D, A, B, C, M, N, K);
    printf("CPU took %fms\n", exe_time_ms);

    // Ejecutamos la prueba de FMA
    printf("[+] Ejecutando prueba DISTRIBUÍDA con split (M, N, K) = (%d, %d, %d)\n",
           M_split, N_split, K_split);
    exe_time_ms = fma_wmma_gpu_distrib(D_DISTRIB, A, B, C,
                                       M, N, K,
                                       M_split, N_split, K_split);
    printf("GPU (distributed, wmma) took %fms\n", exe_time_ms);
    printf("MSE: %f\n", mse(D, D_DISTRIB, M, N));

    errores = allequal(D, D_DISTRIB, M, N, TOL);
    if (errores)
    {
        fprintf(stderr,
                "[WARNING] Existen %d errores entre D y D_DISTRIB! (TOL = %f)\n",
                errores, TOL);

        if (M * N < 100)
        {
            printf("D:\n");
            print_mat(D, M, N);
            printf("\nD_DISTRIB: \n");
            print_mat(D_DISTRIB, M, N);
            printf("\n");
        }
    }

    return 0;
}