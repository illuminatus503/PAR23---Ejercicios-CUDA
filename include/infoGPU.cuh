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

void get_gpu_info(struct info_t *infoGPUs);
