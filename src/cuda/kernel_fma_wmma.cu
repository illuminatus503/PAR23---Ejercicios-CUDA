/**
 * El código de este kernel está basado en el código del artículo de nvidia.developer:
 *
 *      Programming Tensor Cores in CUDA 9 / Programmatic Access to Tensor Cores in CUDA 9.0 -->
 *      https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/#:~:text=Programmatic%20Access%20to%20Tensor%20Cores%20in%20CUDA%209.0
 *
 *      WMMA_TensorCores_Examples (By WZSH)
 *      https://github.com/wzsh/wmma_tensorcore_sample/blob/master/matrix_wmma/matrix_wmma/main.cu
 *
 * Hay que compilar este kernel con soporte para SM 75 o superior. Por ejemplo, con -arch sm_75.
 */

#include <mma.h>
using namespace nvcuda; // compilar con

#include "../../include/cuda/kernel_fma.cuh"

__global__ void cuda_fma_wmma(half *A, half *B, float *C, float *D, int M, int N, int K)
{
    int a_col, a_row, b_col, b_row, c_col, c_row;
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int iy = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments for A, B, accumulator for AB and C
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the AB fragment
    wmma::fill_fragment(ab_frag, 0.0f);

    // Perform AB = A*B
    a_row = ix * WMMA_M;
    b_row = iy * WMMA_N;
    for (int k = 0; k < K; k += WMMA_K)
    {
        a_col = k;
        b_col = k;

        if (a_row < M && a_col < K && b_row < K && b_col < N)
        {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, A + a_col + a_row * K, K);
            wmma::load_matrix_sync(b_frag, B + b_col + b_row * K, K);

            // Perform the matrix multiplication
            wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
        }
    }

    // Perform D = AB + C
    c_col = b_row;
    c_row = a_row;
    if (c_row < M && c_col < N)
    {
        wmma::load_matrix_sync(c_frag, C + c_col + c_row * N, N, wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(D + c_col + c_row * N, c_frag, N, wmma::mem_row_major);
    }
}
