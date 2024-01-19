#ifndef __FADD_CUH__
#define __FADD_CUH__

double fma_gpu_global(float *D, const float *A, const float *B, const float *C,
                      const int M, const int N, const int K);

double fma_gpu_shared(float *D, const float *A, const float *B, const float *C,
                      const int M, const int N, const int K);

double fma_wmma_gpu(float *A_, float *B_, float *C_, float *D,
                    int M, int N, int K);

#endif