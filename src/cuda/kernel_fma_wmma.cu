/**
 * El código de este kernel está basado en el código del artículo de nvidia.developer:
 *
 *      Programming Tensor Cores in CUDA 9 / Programmatic Access to Tensor Cores in CUDA 9.0 -->
 *      https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/#:~:text=Programmatic%20Access%20to%20Tensor%20Cores%20in%20CUDA%209.0
 *
 * Hay que compilar este kernel con soporte para SM 75 o superior. Por ejemplo, con -arch=sm_75.
 */

#include "../../include/cuda/kernel_fma.cuh"

#include <mma.h>
using namespace nvcuda; // compilar con

__global__ void cuda_fma_wmma(half *a, half *b, float *c,
                              int M, int N, int K,
                              float alpha, float beta)
{
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < K; i += WMMA_K)
    {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N)
        {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N)
    {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

#pragma unroll
        for (int i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}
