/**
 * @brief Kernel 
 * 
 * @param A_ 
 * @param B_ 
 * @param C_ 
 * @param D 
 * @param N 
 * @param M 
 * @param P 
 * @return __global__ 
 */
__global__ void cuda_fma_global(float *A_, float *B_, float *C_, float *D,
                                int N, int M, int P);
