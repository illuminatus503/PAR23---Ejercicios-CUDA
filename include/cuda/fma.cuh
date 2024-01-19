#ifndef __FADD_CUH__
#define __FADD_CUH__

double fma_gpu_global(float *D, const float *A, const float *B, const float *C,
                      const int M, const int N, const int K);

double fma_gpu_shared(float *D, const float *A, const float *B, const float *C,
                      const int M, const int N, const int K);

double fma_wmma_gpu(float *A_, int N1, int M1,
                    float *B_, int N2, int M2,
                    float *C_, int N3, int M3,
                    float *D, int N, int M);

#endif