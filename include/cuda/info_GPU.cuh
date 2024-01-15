#ifndef __INFO_GPU__
#define __INFO_GPU__

struct infoGPU_t
{
    int gpu_id;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int multiProcessorCount;
    unsigned long memFree, memTotal;
    unsigned long sharedMemPerBlock;
};

struct info_t
{
    int num_gpus;
    struct infoGPU_t *infoGPU;
};

/**
 * @brief Get information about the group of GPUs of the system.
 *
 * @param infoGPUs A list of infoGPU_t struct (information of each GPU)
 */
void get_gpu_info(struct info_t *infoGPUs);

/**
 * @brief Print information about the GPU system to stdout.
 *
 */
void print_gpu_info();

#endif
