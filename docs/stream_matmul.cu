#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h> // Para time()

#define M 256
#define N 256
#define K 256
#define NUM_STREAMS 4

// Kernel simple para multiplicación de matrices
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numARows && col < numBColumns)
    {
        float sum = 0.0;
        for (int i = 0; i < numAColumns; ++i)
        {
            sum += A[row * numAColumns + i] * B[i * numBColumns + col];
        }
        C[row * numBColumns + col] = sum;
    }
}

void matrixMultiplyCPU(float *A, float *B, float *C, int numARows, int numAColumns, int numBColumns)
{
    for (int row = 0; row < numARows; ++row)
    {
        for (int col = 0; col < numBColumns; ++col)
        {
            float sum = 0.0;
            for (int i = 0; i < numAColumns; ++i)
            {
                sum += A[row * numAColumns + i] * B[i * numBColumns + col];
            }
            C[row * numBColumns + col] = sum;
        }
    }
}

float calculateMSE(float *matrix1, float *matrix2, int numRows, int numCols)
{
    float sumError = 0;
    for (int i = 0; i < numRows; ++i)
    {
        for (int j = 0; j < numCols; ++j)
        {
            float diff = matrix1[i * numCols + j] - matrix2[i * numCols + j];
            sumError += diff * diff;
        }
    }
    return sumError / (numRows * numCols);
}

int main()
{
    // Punteros a las matrices en el host
    float *h_A, *h_B, *h_C;
    // Punteros a las matrices en el device
    float *d_A, *d_B, *d_C;

    // Asignar memoria en el host
    h_A = (float *)malloc(sizeof(float) * M * K);
    h_B = (float *)malloc(sizeof(float) * K * N);
    h_C = (float *)malloc(sizeof(float) * M * N);

    // Inicializar la semilla de los números aleatorios
    srand(time(NULL));

    // Inicializar las matrices con valores aleatorios
    for (int i = 0; i < M * K; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX; // Números entre 0 y 1
    }
    for (int i = 0; i < K * N; ++i)
    {
        h_B[i] = rand() / (float)RAND_MAX; // Números entre 0 y 1
    }
    for (int i = 0; i < M * N; ++i)
    {
        h_C[i] = 0; // Inicializar la matriz C con ceros
    }

    // Multiplicación de matrices en CPU
    float *h_C_CPU = (float *)malloc(sizeof(float) * M * N);
    matrixMultiplyCPU(h_A, h_B, h_C_CPU, M, K, N);

    // Asignar memoria en el device
    cudaMalloc(&d_A, sizeof(float) * M * K);
    cudaMalloc(&d_B, sizeof(float) * K * N);
    cudaMalloc(&d_C, sizeof(float) * M * N);

    // Crear streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Tamaño de cada parte (asumiendo que M es divisible por NUM_STREAMS por simplicidad)
    int subM = M / NUM_STREAMS;

    // Copiar datos a la memoria del device y ejecutar el kernel en cada stream
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        int offset = i * subM * K;
        cudaMemcpyAsync(d_A + offset, h_A + offset, sizeof(float) * subM * K, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B, h_B, sizeof(float) * K * N, cudaMemcpyHostToDevice, streams[i]);

        dim3 dimGrid((N + 15) / 16, (subM + 15) / 16, 1);
        dim3 dimBlock(16, 16, 1);
        matrixMultiply<<<dimGrid, dimBlock, 0, streams[i]>>>(d_A + offset, d_B, d_C + offset, subM, K, N);

        cudaMemcpyAsync(h_C + offset, d_C + offset, sizeof(float) * subM * N, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Sincronizar los streams
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        cudaStreamSynchronize(streams[i]);
    }

    // Liberar recursos
    for (int i = 0; i < NUM_STREAMS; ++i)
    {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Calcular el MSE entre los resultados de la CPU y la GPU
    float mse = calculateMSE(h_C_CPU, h_C, M, N);
    printf("Mean Squared Error between CPU and GPU results: %f\n", mse);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_CPU);

    return 0;
}
