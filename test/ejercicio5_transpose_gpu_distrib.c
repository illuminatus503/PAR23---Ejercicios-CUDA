#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../include/utils.h"

#include "../include/cuda/linalg.cuh"
#include "../include/cuda/utils.cuh"

#define M 100
#define N 60

#define M_split 10 // si M_split => M, entonces, se toman filas de 1 en 1
#define N_split 1 // si N_split => N, entonces, se toman columnas de 1 en 1

#define TOL (float)1e-4

int main()
{
    float A[M * N], A_gpu[M * N], A_gpu_2[M * N];
    double exe_time_ms;
    int errores;

    // Generamos la matriz A
    rand_init(A, M, N);

    // Ejecutamos la prueba de TRASPUESTA
    printf("[+] Ejecutando prueba TRASPUESTA DISTRIBUÍDA con split (M, N) = (%d, %d)\n",
           M_split, N_split);

    // exe_time_ms = transpose_cuda(A_gpu, A, M, N);
    // exe_time_ms += transpose_cuda(A_gpu, A_gpu, N, M);

    exe_time_ms = transpose_distributed(A_gpu, A, M, N, M_split, N_split);
    exe_time_ms += transpose_distributed(A_gpu_2, A_gpu, N, M, N_split, M_split);
    exe_time_ms /= 2.0f;

    printf("GPU took %fms\n", exe_time_ms);
    printf("MSE: %f\n", mse(A, A_gpu_2, M, N));

    errores = allequal(A, A_gpu_2, M, N, TOL);
    if (errores)
    {
        fprintf(stderr,
                "[WARNING] Existen %d errores! (TOL = %f)\n",
                errores, TOL);

        if (M * N < 100)
        {
            print_mat(A, M, N);
            printf("\n\n");
            print_mat(A_gpu_2, M, N);
            printf("\n");
        }
    }

    return 0;
}