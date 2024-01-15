#include "../../include/cuda/utils.cuh"

#include <stdio.h>
#include <stdlib.h>

void load_gpu_info(struct info_t *gpu_array)
{
    int i, num_gpus, gpu_id;
    cudaDeviceProp devProp;

    // Determinar el número de GPUs del sistema
    cudaGetDeviceCount(&num_gpus);
    gpu_array->num_gpus = num_gpus;

    // Reservamos espacio para los structs con info sobre cada GPU
    gpu_array->infoGPU = (struct infoGPU_t *)malloc(num_gpus * sizeof(struct infoGPU_t));

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++)
    {
        // Cambiamos a la i-esima GPU
        cudaSetDevice(gpu_id);

        // Guardamos el último estado de la memoria.
        cudaMemGetInfo(&gpu_array->infoGPU[gpu_id].free_mem,
                       &gpu_array->infoGPU[gpu_id].total_mem);

        // Guardamos propiedades generales de la GPU
        // * Identificador del sistema (0, 1, ...)
        // * Número máximo de hilos por bloque
        // * Número de SM
        // * Tamaño de la memoria compartida por bloque
        // * Tamaño máximo en hilos por cada dimensión de un bloque
        // * Número máximo de bloques por cada dimensión del grid
        cudaGetDeviceProperties(&devProp, gpu_id);
        gpu_array->infoGPU[gpu_id].gpu_id = gpu_id;
        gpu_array->infoGPU[gpu_id].maxThreadsPerBlock = devProp.maxThreadsPerBlock;
        gpu_array->infoGPU[gpu_id].multiProcessorCount = devProp.multiProcessorCount;
        gpu_array->infoGPU[gpu_id].sharedMemPerBlock = devProp.sharedMemPerBlock;
        for (i = 0; i < 3; i++)
        {
            gpu_array->infoGPU[gpu_id].maxThreadsDim[i] = devProp.maxThreadsDim[i];
            gpu_array->infoGPU[gpu_id].maxGridSize[i] = devProp.maxGridSize[i];
        }
    }
}

void update_gpu_info(struct info_t *gpu_array, long gpu_id)
{
    int _gpu_id;

    if (gpu_array == (void *)0)
    {
        fprintf(stderr, "[ERROR] GPU property array is empty!! \n");
        exit(1);
    }

    if (gpu_id == -1)
    {
        for (_gpu_id = gpu_id; _gpu_id < gpu_array->num_gpus; _gpu_id++)
        {
            // Cambiamos a la i-esima GPU
            cudaSetDevice(_gpu_id);

            // Guardamos el último estado de la memoria.
            cudaMemGetInfo(&gpu_array->infoGPU[_gpu_id].free_mem,
                           &gpu_array->infoGPU[_gpu_id].total_mem);
        }
    }
    else
    {
        // Cambiamos a la i-esima GPU
        cudaSetDevice(gpu_id);

        // Guardamos el último estado de la memoria.
        cudaMemGetInfo(&gpu_array->infoGPU[gpu_id].free_mem,
                       &gpu_array->infoGPU[gpu_id].total_mem);
    }
}

void print_gpu_info(struct info_t *gpu_array)
{
    int gpu_id;
    struct infoGPU_t device_prop;

    if (gpu_array == NULL)
    {
        fprintf(stderr, "[ERROR] GPU property array is empty!! \n");
        exit(1);
    }

    printf("==== GPU Analysis ====\n");
    printf("Total Number of GPUs: %d\n", gpu_array->num_gpus);
    printf("----------------------------------------\n");

    for (gpu_id = 0; gpu_id < gpu_array->num_gpus; gpu_id++)
    {
        cudaSetDevice(gpu_id); // Set the current GPU
        device_prop = gpu_array->infoGPU[gpu_id];

        printf("GPU ID: %i\n", gpu_id);
        printf("Free Memory: %lu MB / Total Memory: %lu MB\n",
               device_prop.free_mem / 1048576, device_prop.total_mem / 1048576);
        printf("Streaming Multiprocessors (SM): %i\n", device_prop.multiProcessorCount);
        printf("Shared Memory per Block: %lu KB\n", device_prop.sharedMemPerBlock / 1024);
        printf("Max Threads per Block: %i\n", device_prop.maxThreadsPerBlock);
        printf("Max Threads Dimensions: x = %i, y = %i, z = %i\n",
               device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
        printf("Max Grid Size: x = %i, y = %i, z = %i\n",
               device_prop.maxGridSize[0], device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
        printf("----------------------------------------\n");
    }
    printf("==== End of GPU Analysis ====\n\n");
}

void clean_gpu_info(struct info_t *gpu_array)
{
    if (gpu_array == NULL)
    {
        fprintf(stderr, "[ERROR] GPU property array is empty!! \n");
        exit(1);
    }

    free(gpu_array->infoGPU);
}
