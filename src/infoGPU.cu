#include "../include/infoGPU.cuh"

void get_gpu_info(struct info_t *infoGPUs)
{
    int i, num_gpus, gpu_id;
    cudaDeviceProp devProp;

    cudaGetDeviceCount(&num_gpus);
    infoGPUs->num_gpus = num_gpus;
    infoGPUs->infoGPU = (struct infoGPU_t *)malloc(num_gpus * sizeof(struct infoGPU_t));
    for (gpu_id = 0; gpu_id < num_gpus; gpu_id++)
    {
        cudaSetDevice(gpu_id);
        cudaMemGetInfo(&infoGPUs->infoGPU[gpu_id].memFree, &infoGPUs->infoGPU[gpu_id].memTotal);
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
