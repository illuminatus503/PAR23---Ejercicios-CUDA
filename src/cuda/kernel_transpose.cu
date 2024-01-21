#include <cuda.h>
#include <cuda_runtime.h>

#include "../../include/cuda/kernel_linalg.cuh"

__global__ void cuda_transpose(float *odata, const float *idata, const int width, const int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 para evitar conflictos de bancos en memoria compartida

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index = xIndex + (yIndex)*width;

    // Cargar la matriz de entrada a la memoria compartida
    if (xIndex < width && yIndex < height)
    { // Verificar límites
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            if (yIndex + j < height)
            {
                tile[threadIdx.y + j][threadIdx.x] = idata[index + j * width];
            }
        }
    }

    __syncthreads();

    // Calcular índices para la transposición
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    index = xIndex + (yIndex)*height; // Cambiar width por height en el cálculo del índice

    // Escribir la matriz transpuesta a la memoria global
    if (xIndex < height && yIndex < width)
    {
        for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        {
            if (yIndex + j < width)
            {
                odata[index + j * height] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}
