#ifndef __KERNEL_LINALG_CUH__
#define __KERNEL_LINALG_CUH__

#define TILE_DIM 32
#define BLOCK_ROWS 8

/**
 * @brief Efficient transpose in CUDA. Taken from
 * https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
 *
 * @param odata float M x N
 * @param idata float N x M
 * @param M
 * @param N
 */
__global__ void cuda_transpose(float *odata, float *idata, const int M, const int N);

#endif