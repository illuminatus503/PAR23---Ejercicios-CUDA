#ifndef __CUDA_UTILS__
#define __CUDA_UTILS__

struct infoGPU_t
{
    int gpu_id;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int multiProcessorCount;
    size_t free_mem, total_mem;
    size_t sharedMemPerBlock;
};

struct info_t
{
    int num_gpus;
    struct infoGPU_t *infoGPU;
};

/**
 * @brief Get information about the group of GPUs of the system.
 *
 * @param gpu_array An array of infoGPU_t struct (information of each GPU)
 */
void load_gpu_info(struct info_t *gpu_array);

/**
 * @brief Update memory allocation information from each GPU, at runtime.
 *
 * @param gpu_array An array of infoGPU_t struct (information of each GPU)
 * @param gpu_id gpu_id >= 0 (long) of the gpu to update. If gpu_id == -1, then, update all.
 */
void update_gpu_info(struct info_t *gpu_array, long gpu_id);

/**
 * @brief Print runtime information about the current status of the system.
 *
 * @param gpu_array An array of infoGPU_t struct (information of each GPU)
 */
void print_gpu_info(struct info_t *gpu_array);

/**
 * @brief Deallocate the array of information.
 *
 * @param gpu_array An array of infoGPU_t struct (information of each GPU)
 */
void clean_gpu_info(struct info_t *gpu_array);

#endif