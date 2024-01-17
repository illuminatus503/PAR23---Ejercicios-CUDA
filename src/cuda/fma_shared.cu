#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#include "../../include/cuda/fma.cuh"
#include "../../include/cuda/kernel_fma.cuh"

#include "../../include/utils.h"
#include "../../include/cuda/error.cuh"
#include "../../include/cuda/utils.cuh"

double __fma_shared_gpu(float *A_, float *B_, float *C_, float *D,
                        int N, int M, int P,
                        struct info_t *gpu_array)
{
    size_t gpu_id;
    infoGPU_t device_prop;
    cudaEvent_t start, stop;
    float exe_time_ms = 0.0;

    // Calculamos el espacio necesario en bytes para las matrices
    float *d_A, *d_B, *d_C, *d_D;
    const size_t size_A = N * M * sizeof(float);  // Matriz A
    const size_t size_B = M * P * sizeof(float);  // Matriz B
    const size_t size_CD = N * P * sizeof(float); // Matrices C y D

    // TODO múltiples dispositivos: por defecto, 0
    gpu_id = 0;
    cudaSetDevice(gpu_id);
    update_gpu_info(gpu_array, gpu_id); // actualizamos la información sobre mem.
    device_prop = gpu_array->infoGPU[gpu_id];

    // Incializamos la medición de tiempo mediante eventos
    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    // Medimos la memoria disponible, total del dispositivo en este
    // momento:
    printf("[+] Memoria disponible en gpu%zu: %zu MB disponibles de %zu MB\n",
           gpu_id,
           device_prop.free_mem / 1048576,
           device_prop.total_mem / 1048576);

    if (device_prop.free_mem < (size_A + size_B + 2 * size_CD))
    {
        fprintf(stderr,
                "[ERROR] No hay suficiente memoria disponible en gpu%zu: son necesarios %zu MB\n",
                gpu_id,
                (size_A + size_B + 2 * size_CD) / 1048576);
        exit(1);
    }

    // Reservamos memoria para las matrices en el dispositivo
    gpuErrchk(cudaMalloc((void **)&d_A, size_A));
    gpuErrchk(cudaMalloc((void **)&d_B, size_B));
    gpuErrchk(cudaMalloc((void **)&d_C, size_CD));
    gpuErrchk(cudaMalloc((void **)&d_D, size_CD));

    // Copiamos los datos necesarios para la operación: matrices A, B y C
    gpuErrchk(cudaMemcpy((void *)d_A, (const void *)A_, size_A, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_B, (const void *)B_, size_B, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((void *)d_C, (const void *)C_, size_CD, cudaMemcpyHostToDevice));

    // Asegúrate de que el número de hilos por bloque no sea mayor que el máximo permitido
    dim3 threadsPerBlock(THR_PER_BLOCK, THR_PER_BLOCK);
    printf("[+] Lanzando 1024 hilos en warp: (%u x %u x %u)\n",
           threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);

    // Calcula el número de bloques necesarios para cubrir todas las operaciones
    dim3 blocksPerGrid((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    printf("[+] Lanzando el kernel en un grid de %u x %u x %u bloques\n",
           blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
    gpuErrchk(cudaEventRecord(start));
    cuda_fma_shared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N, M, P);
    cudaCheckError(); // Check error after execution
    gpuErrchk(cudaEventRecord(stop));
    printf("[+] Recuperando datos del dispositivo...\n");

    // Copy data from device array to host array
    gpuErrchk(cudaMemcpy((void *)D, (const void *)d_D, size_CD, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaEventSynchronize(stop));
    gpuErrchk(cudaEventElapsedTime(&exe_time_ms, start, stop));

    /**
     * Free CUDA mem.
     */
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));
    gpuErrchk(cudaFree(d_D));

    return (double)exe_time_ms;
}

double fma_shared_gpu(float *A_, int N1, int M1,
                      float *B_, int N2, int M2,
                      float *C_, int N3, int M3,
                      float *D, int N, int M,
                      struct info_t *gpu_array)
{
    if (!matrix_checkdims(N1, M1, N2, M2, N3, M3, N, M))
    {
        fprintf(stderr,
                "[DimError] La dimensiones de las matrices no coinciden: A(%d x %d) · B(%d x %d) + C(%d x %d) = D(%d x %d)\n",
                N1, M1, N2, M2, N3, M3, N, M);
        return 0.0; // Asum. que el checkeo no añade sobrecostes
    }

    return __fma_shared_gpu(A_, B_, C_, D, N, M1, M, gpu_array);
}
