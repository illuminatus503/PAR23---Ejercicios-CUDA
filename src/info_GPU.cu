#include <stdio.h>

#include "../include/info_GPU.cuh"

void get_gpu_info(struct info_t *infoGPUs)
{
    int i, num_gpus, gpu_id;
    cudaDeviceProp devProp;

    // Determinar el número de GPUs del sistema
    cudaGetDeviceCount(&num_gpus);
    infoGPUs->num_gpus = num_gpus;

    // Reservamos espacio para los structs con info sobre cada GPU
    infoGPUs->infoGPU = (struct infoGPU_t *)malloc(num_gpus * sizeof(struct infoGPU_t));

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++)
    {
        // Cambiamos a la i-esima GPU
        cudaSetDevice(gpu_id);

        // Guardamos información sobre la información disponible, total
        cudaMemGetInfo(&infoGPUs->infoGPU[gpu_id].memFree,
                       &infoGPUs->infoGPU[gpu_id].memTotal);

        // Guardamos propiedades generales de la GPU
        // * Identificador del sistema (0, 1, ...)
        // * Número máximo de hilos por bloque
        // * Número de SM
        // * Tamaño de la memoria compartida por bloque
        // * Tamaño máximo en hilos por cada dimensión de un bloque
        // * Número máximo de bloques por cada dimensión del grid
        cudaGetDeviceProperties(&devProp, gpu_id);
        infoGPUs->infoGPU[gpu_id].gpu_id = gpu_id;
        infoGPUs->infoGPU[gpu_id].maxThreadsPerBlock = devProp.maxThreadsPerBlock;
        infoGPUs->infoGPU[gpu_id].multiProcessorCount = devProp.multiProcessorCount;
        infoGPUs->infoGPU[gpu_id].sharedMemPerBlock = devProp.sharedMemPerBlock;
        for (i = 0; i < 3; i++)
        {
            infoGPUs->infoGPU[gpu_id].maxThreadsDim[i] = devProp.maxThreadsDim[i];
            infoGPUs->infoGPU[gpu_id].maxGridSize[i] = devProp.maxGridSize[i];
        }
    }
}

void print_gpu_info()
{
    int num_gpus, gpu_id;
    cudaDeviceProp devProp;
    size_t free_mem, total_mem;

    // Determinar el número de GPUs del sistema
    cudaGetDeviceCount(&num_gpus);

    printf(" ** GPU analysis ** \n");
    printf(" - NUM GPUS: %d\n", num_gpus);

    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++)
    {
        // Cambiamos a la i-esima GPU
        cudaSetDevice(gpu_id);

        // Guardamos información sobre la información disponible, total
        cudaMemGetInfo(&free_mem, &total_mem);
        printf(" ! GPU ID: %i  [Free mem. (MB)= %lu / Total mem. (MB) = %lu]\n",
               gpu_id, free_mem / 1048576, total_mem / 1048576);

        // Guardamos propiedades generales de la GPU
        // * Identificador del sistema (0, 1, ...)
        // * Número máximo de hilos por bloque
        // * Número de SM
        // * Tamaño de la memoria compartida por bloque
        // * Tamaño máximo en hilos por cada dimensión de un bloque
        // * Número máximo de bloques por cada dimensión del grid
        cudaGetDeviceProperties(&devProp, gpu_id);
        printf(" ** Num. Streaming Multiprocessors (SM) = %i\n", devProp.multiProcessorCount);
        printf(" ** Shared memory per block (KB) = %lu\n", devProp.sharedMemPerBlock / 1024);
        printf(" ** Max. threads per block = %i | Per dim. {max. x, max. y, max. z} = {%i, %i, %i}\n",
               devProp.maxThreadsPerBlock,
               devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf(" ** Max. grid size: {max. x, max. y, max. z} = {%i, %i, %i}\n",
               devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
    }
    printf(" ** END **\n\n");
}
