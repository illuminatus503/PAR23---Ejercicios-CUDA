#include <cuda.h>
// #include <cuda_runtime.h>

#include <stdio.h>

#include "../include/codeGPU.cuh"

#define THR_PER_BLOCK 1024

__global__ void cuda_vec_add(float *A, float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

double add_vectors_GPU(float *A, float *B, float *C, size_t N)
{
    cudaEvent_t start, stop;
    float *d_A, *d_B, *d_C;
    float milliseconds = 0;
    int thr_per_blk, blk_in_grid;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    gpuErrchk(cudaMalloc(&d_A, N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_B, N * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_C, N * sizeof(float)));

    // Copy data from host arrays A and B to device arrays d_A and d_B
    gpuErrchk(cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice));

    // Set execution configuration parameters
    //      thr_per_blk: number of CUDA threads per grid block
    //      blk_in_grid: number of blocks in grid
    thr_per_blk = THR_PER_BLOCK;
    blk_in_grid = ceil((float)N / thr_per_blk);

    // Launch kernel
    gpuErrchk(cudaEventRecord(start));
    cuda_vec_add<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C, N);
    gpuErrchk(cudaEventRecord(stop));

    // Copy data from device array d_C to host array C
    cudaMemcpy(C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return (milliseconds);
}
