#ifndef __CODECPU_H__
#define __CODECPU_H__

#include <stdlib.h>

double timing_CPU(struct timespec begin, struct timespec end);
void init_vectors(float *A, float *B, int N);
double add_vectors_CPU(float *A, float *B, float *C, int N);
double prod_vectors_CPU(float *A, float *B, float *C, int N);
double dot_prod_vectors_CPU(float *A, float *B, float *C, int N);
void print_vector(float *C, int N);

#endif