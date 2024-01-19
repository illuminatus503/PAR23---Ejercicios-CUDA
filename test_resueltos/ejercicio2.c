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
    double time_naive_gpu, time_cpu;
    float *A, *B, *C, *D_cpu, *D_gpu;

    int N, M, P;
    int i;

    double times_cpu_var[NUM_SIZES_VAR], times_gpu_var[NUM_SIZES_VAR];
    double infty_dists_var[NUM_SIZES_VAR], mses_var[NUM_SIZES_VAR];
    double memory_required_var[NUM_SIZES_VAR];

    double times_cpu_stress[NUM_SIZES_STRESS], times_gpu_stress[NUM_SIZES_STRESS];
    double infty_dists_stress[NUM_SIZES_STRESS], mses_stress[NUM_SIZES_STRESS];
    double memory_required_stress[NUM_SIZES_STRESS];

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

        // Inicialización y ejecución de las pruebas...
        // Init. las matrices en mem. din.
        if (i == 0)
        {
            A = (float *)malloc(N * M * sizeof(float));
            B = (float *)malloc(M * P * sizeof(float));
            C = (float *)malloc(N * P * sizeof(float));
            D_cpu = (float *)malloc(N * P * sizeof(float));
            D_gpu = (float *)malloc(N * P * sizeof(float));
        }
        else
        {
            A = (float *)realloc(A, N * M * sizeof(float));
            B = (float *)realloc(B, M * P * sizeof(float));
            C = (float *)realloc(C, N * P * sizeof(float));
            D_cpu = (float *)realloc(D_cpu, N * P * sizeof(float));
            D_gpu = (float *)realloc(D_gpu, N * P * sizeof(float));
        }

        memory_required_var[i] = size_of_mat(N, M, P); // in MB
        gen_matrices(N, M, P, A, B, C);

        // FMA en CPU
        time_cpu = fma_cpu(A, N, M, B, M, P, C, N, P, D_cpu, N, P);

        // FMA en GPU (memoria global)
        time_naive_gpu = fma_gpu_global(A, N, M, B, M, P, C, N, P, D_gpu, N, P, &gpu_array);

        // Guardar métricas
        times_cpu_var[i] = time_cpu;
        times_gpu_var[i] = time_naive_gpu;
        infty_dists_var[i] = matrix_infty_dist(D_cpu, D_gpu, N, P);
        mses_var[i] = mse(D_cpu, D_gpu, N, P);
    }

    printf("[!] LANZANDO PRUEBA DE ESTRÉS\n");

    for (i = 0; i < NUM_SIZES_STRESS; i++)
    {
        N = sizes_stress[i][0];
        M = sizes_stress[i][1];
        P = sizes_stress[i][2];

        // Inicialización y ejecución de las pruebas...
        // Init. las matrices en mem. din.
        A = (float *)realloc(A, N * M * sizeof(float));
        B = (float *)realloc(B, M * P * sizeof(float));
        C = (float *)realloc(C, N * P * sizeof(float));
        D_cpu = (float *)realloc(D_cpu, N * P * sizeof(float));
        D_gpu = (float *)realloc(D_gpu, N * P * sizeof(float));

        memory_required_stress[i] = size_of_mat(N, M, P); // in MB
        gen_matrices(N, M, P, A, B, C);

        // FMA en CPU
        time_cpu = fma_cpu(A, N, M, B, M, P, C, N, P, D_cpu, N, P);

        // FMA en GPU (memoria global)
        time_naive_gpu = fma_gpu_global(A, N, M, B, M, P, C, N, P, D_gpu, N, P, &gpu_array);

        // Guardar métricas
        times_cpu_stress[i] = time_cpu;
        times_gpu_stress[i] = time_naive_gpu;
        infty_dists_stress[i] = matrix_infty_dist(D_cpu, D_gpu, N, P);
        mses_stress[i] = mse(D_cpu, D_gpu, N, P);
    }

    // Guardamos los resultados en un fichero
    FILE *file = fopen("ejercicio2_resultados.csv", "w");
    if (file == NULL)
    {
        perror("Error al abrir el archivo");
        return 1;
    }

    // Escribir la cabecera del CSV
    fprintf(file,
            "Tipo de Prueba,N,M,P,Tamaño de Memoria (MB),Tiempo CPU (ms),Tiempo GPU (ms),Distancia Infinita,MSE\n");

    // Impresión de resultados
    printf("----------------------------------------------------------------------------------------\n");
    printf("Tipo de Prueba | N    | M    | P    | Tamaño de Memoria (MB) | Tiempo CPU (ms) | Tiempo GPU (ms) | Distancia Infinita | MSE\n");
    printf("----------------------------------------------------------------------------------------\n");
    for (i = 0; i < NUM_SIZES_VAR; i++)
    {
        fprintf(file,
                "Variaciones,%d,%d,%d,%f,%f,%f,%f,%f\n",
                sizes_var[i][0], sizes_var[i][1], sizes_var[i][2],
                memory_required_var[i],
                times_cpu_var[i], times_gpu_var[i],
                infty_dists_var[i], mses_var[i]);
        printf("Variaciones    | %4d | %4d | %4d | %15.3f | %15.3f | %15.3f | %20.3f | %f\n",
               sizes_var[i][0], sizes_var[i][1], sizes_var[i][2],
               memory_required_var[i],
               times_cpu_var[i], times_gpu_var[i],
               infty_dists_var[i], mses_var[i]);
    }

    for (i = 0; i < NUM_SIZES_STRESS; i++)
    {
        fprintf(file,
                "Estrés,%d,%d,%d,%f,%f,%f,%f,%f\n",
                sizes_stress[i][0], sizes_stress[i][1], sizes_stress[i][2],
                memory_required_stress[i],
                times_cpu_stress[i], times_gpu_stress[i],
                infty_dists_stress[i], mses_stress[i]);
        printf("Estrés         | %4d | %4d |%4d | %15.3f | %15.3f | %15.3f | %20.3f | %f\n",
               sizes_stress[i][0], sizes_stress[i][1], sizes_stress[i][2],
               memory_required_stress[i],
               times_cpu_stress[i], times_gpu_stress[i],
               infty_dists_stress[i], mses_stress[i]);
    }

    printf("-----------------------------------------------------------\n");

    fclose(file);

    // Liberar memoria dinámica reservada
    free(A);
    free(B);
    free(C);
    free(D_cpu);
    free(D_gpu);

    clean_gpu_info(&gpu_array);

    return 0;
}
