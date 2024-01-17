#include <stdio.h>
#include <stdlib.h>

#include "../include/utils.h"

#include "../include/fma.h"
#include "../include/cuda/fma.cuh"
#include "../include/cuda/utils.cuh"

#define NUM_SIZES_VAR 3
#define NUM_SIZES_STRESS 3

#define size_of_mat(N, M, P) ((N * M + M * P + 2 * N * P) * sizeof(float)) / 1048576.0

int main()
{
    double time_naive_gpu, time_shared_gpu;
    float *A, *B, *C, *D_gpu, *D_gpu_shared;

    int N, M, P;
    int i;

    double times_naive_gpu_var[NUM_SIZES_VAR], times_shared_gpu_var[NUM_SIZES_VAR];
    double times_naive_gpu_stress[NUM_SIZES_STRESS], times_shared_gpu_stress[NUM_SIZES_STRESS];
    double memory_required_var[NUM_SIZES_VAR], memory_required_stress[NUM_SIZES_STRESS];
    double mse_var[NUM_SIZES_VAR], mse_stress[NUM_SIZES_STRESS];
    double infty_dist_var[NUM_SIZES_VAR], infty_dist_stress[NUM_SIZES_STRESS];

    int sizes_var[][3] = {
        {100, 100, 100},
        {500, 500, 500},
        {1000, 1000, 1000},
    };

    int sizes_stress[][3] = {
        {2000, 2000, 2000},
        {4000, 4000, 4000},
        {10000, 10000, 10000},
    };

    struct info_t gpu_array;
    load_gpu_info(&gpu_array);

    printf("[!] LANZANDO PRUEBA DE VARIACIONES\n");

    for (i = 0; i < NUM_SIZES_VAR; i++)
    {
        N = sizes_var[i][0];
        M = sizes_var[i][1];
        P = sizes_var[i][2];

        A = (i == 0) ? (float *)malloc(N * M * sizeof(float)) : (float *)realloc(A, N * M * sizeof(float));
        B = (i == 0) ? (float *)malloc(M * P * sizeof(float)) : (float *)realloc(B, M * P * sizeof(float));
        C = (i == 0) ? (float *)malloc(N * P * sizeof(float)) : (float *)realloc(C, N * P * sizeof(float));
        D_gpu = (i == 0) ? (float *)malloc(N * P * sizeof(float)) : (float *)realloc(D_gpu, N * P * sizeof(float));
        D_gpu_shared = (i == 0) ? (float *)malloc(N * P * sizeof(float)) : (float *)realloc(D_gpu_shared, N * P * sizeof(float));

        memory_required_var[i] = size_of_mat(N, M, P); // in MB
        gen_matrices(N, M, P, A, B, C);

        time_naive_gpu = fma_global_gpu(A, N, M, B, M, P, C, N, P, D_gpu, N, P, &gpu_array);
        time_shared_gpu = fma_shared_gpu(A, N, M, B, M, P, C, N, P, D_gpu_shared, N, P, &gpu_array);

        times_naive_gpu_var[i] = time_naive_gpu;
        times_shared_gpu_var[i] = time_shared_gpu;
        mse_var[i] = mse(D_gpu, D_gpu_shared, N, P);
        infty_dist_var[i] = matrix_infty_dist(D_gpu, D_gpu_shared, N, P);
    }

    printf("[!] LANZANDO PRUEBA DE ESTRÉS\n");

    for (i = 0; i < NUM_SIZES_STRESS; i++)
    {
        N = sizes_stress[i][0];
        M = sizes_stress[i][1];
        P = sizes_stress[i][2];

        A = (float *)realloc(A, N * M * sizeof(float));
        B = (float *)realloc(B, M * P * sizeof(float));
        C = (float *)realloc(C, N * P * sizeof(float));
        D_gpu = (float *)realloc(D_gpu, N * P * sizeof(float));
        D_gpu_shared = (float *)realloc(D_gpu_shared, N * P * sizeof(float));

        memory_required_stress[i] = size_of_mat(N, M, P); // in MB
        gen_matrices(N, M, P, A, B, C);

        time_naive_gpu = fma_global_gpu(A, N, M, B, M, P, C, N, P, D_gpu, N, P, &gpu_array);
        time_shared_gpu = fma_shared_gpu(A, N, M, B, M, P, C, N, P, D_gpu_shared, N, P, &gpu_array);

        times_naive_gpu_stress[i] = time_naive_gpu;
        times_shared_gpu_stress[i] = time_shared_gpu;
        mse_stress[i] = mse(D_gpu, D_gpu_shared, N, P);
        infty_dist_stress[i] = matrix_infty_dist(D_gpu, D_gpu_shared, N, P);
    }

    // Guardamos los resultados en un fichero
    FILE *file = fopen("ejercicio3_resultados.csv", "w");
    if (file == NULL)
    {
        perror("Error al abrir el archivo");
        return 1;
    }

    // Escribir la cabecera del CSV
    fprintf(file, "Tipo de Prueba,N,M,P,Tamaño de Memoria (MB),Tiempo GPU Naive (ms),Tiempo GPU Shared (ms),Distancia Infinita,MSE\n");

    // Impresión de resultados
    printf("-------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("Tipo de Prueba |   N   |   M   |   P   | Tamaño Mem. (MB) | Tiempo GPU Naive (ms) | Tiempo GPU Shared (ms) | Distancia Infinita | MSE\n");
    printf("-------------------------------------------------------------------------------------------------------------------------------------\n");
    for (i = 0; i < NUM_SIZES_VAR; i++)
    {
        fprintf(file, "Variaciones,%d,%d,%d,%f,%f,%f,%f,%f\n",
                sizes_var[i][0], sizes_var[i][1], sizes_var[i][2],
                memory_required_var[i],
                times_naive_gpu_var[i], times_shared_gpu_var[i],
                infty_dist_var[i], mse_var[i]);
        printf("Variaciones    | %4d | %4d | %4d | %17.3f | %21.3f | %21.3f | %17.3f | %.6f\n",
               sizes_var[i][0], sizes_var[i][1], sizes_var[i][2],
               memory_required_var[i],
               times_naive_gpu_var[i], times_shared_gpu_var[i],
               infty_dist_var[i], mse_var[i]);
    }

    for (i = 0; i < NUM_SIZES_STRESS; i++)
    {
        fprintf(file, "Estrés,%d,%d,%d,%f,%f,%f,%f,%f\n",
                sizes_stress[i][0], sizes_stress[i][1], sizes_stress[i][2],
                memory_required_stress[i],
                times_naive_gpu_stress[i], times_shared_gpu_stress[i],
                infty_dist_stress[i], mse_stress[i]);
        printf("Estrés         | %4d | %4d | %4d | %17.3f | %21.3f | %21.3f | %17.3f | %.6f\n",
               sizes_stress[i][0], sizes_stress[i][1], sizes_stress[i][2],
               memory_required_stress[i],
               times_naive_gpu_stress[i], times_shared_gpu_stress[i],
               infty_dist_stress[i], mse_stress[i]);
    }

    printf("-------------------------------------------------------------------------------------------------------------------------------------\n");

    fclose(file);

    // Liberar memoria dinámica reservada
    free(A);
    free(B);
    free(C);
    free(D_gpu);
    free(D_gpu_shared);

    clean_gpu_info(&gpu_array);

    return 0;
}
