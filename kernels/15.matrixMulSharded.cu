#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

/*

In this snippet, we will learn how to shard a matrix multiplication operation across multiple GPUs.
Sharding is a technique used to divide a large dataset or computation into smaller parts and distribute them across multiple resources.
In this case, we will shard the matrix multiplication operation across multiple GPUs to improve performance.
We will use the matrix multiplication kernel from the previous snippet and shard the input matrices across multiple GPUs.
We will then perform the matrix multiplication operation on each shard using a separate GPU.
Finally, we will combine the results from each GPU to get the final result.

*/

__global__ void matrixMulKernel(
    const float* matrixA, 
    const float* matrixB, 
    float* matrixC, 
    int numRowsA, 
    int numColsA, 
    int numColsB
    ) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRowsA && col < numColsB) {
        float value = 0;
        for (int k = 0; k < numColsA; ++k) {
            value += matrixA[row * numColsA + k] * matrixB[k * numColsB + col];
        }
        matrixC[row * numColsB + col] = value;
    }
}


void matrixMultiplySharded(
    const float* matrixA, 
    const float* matrixB, 
    float* matrixC, 
    int numRowsA, 
    int numColsA, 
    int numColsB
    ) {

    int numGpus;
    cudaGetDeviceCount(&numGpus);
    int rowsPerGpu = numRowsA / numGpus;
    int remainingRows = numRowsA % numGpus;
    float** hostMatrixCChunks = new float*[numGpus];

    for (int gpuIndex = 0; gpuIndex < numGpus; ++gpuIndex) {
        cudaSetDevice(gpuIndex);
        int startRow = gpuIndex * rowsPerGpu;
        int endRow = startRow + rowsPerGpu;

        if (gpuIndex == numGpus - 1) {
            endRow += remainingRows;
        }

        int chunkRows = endRow - startRow;
        size_t sizeAChunk = chunkRows * numColsA * sizeof(float);
        size_t sizeCChunk = chunkRows * numColsB * sizeof(float);

        float *deviceMatrixA, *deviceMatrixB, *deviceMatrixC;

        cudaMalloc(&deviceMatrixA, sizeAChunk);
        cudaMalloc(&deviceMatrixB, numColsA * numColsB * sizeof(float));
        cudaMalloc(&deviceMatrixC, sizeCChunk);

        cudaMemcpy(deviceMatrixA, matrixA + startRow * numColsA, sizeAChunk, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceMatrixB, matrixB, numColsA * numColsB * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim((numColsB + BLOCK_SIZE - 1) / BLOCK_SIZE, (chunkRows + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matrixMulKernel<<<gridDim, blockDim>>>(
            deviceMatrixA, 
            deviceMatrixB, 
            deviceMatrixC, 
            chunkRows, 
            numColsA, 
            numColsB);

        cudaMemcpy(matrixC + startRow * numColsB, deviceMatrixC, sizeCChunk, cudaMemcpyDeviceToHost);

        cudaFree(deviceMatrixA);
        cudaFree(deviceMatrixB);
        cudaFree(deviceMatrixC);
    }

    delete[] hostMatrixCChunks;
}


void fillMatrix(float* matrix, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            matrix[row * cols + col] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
}


int main() {
    int numRowsA = 1024;
    int numColsA = 512;
    int numRowsB = numColsA;
    int numColsB = 1024;

    size_t sizeA = numRowsA * numColsA * sizeof(float);
    size_t sizeB = numRowsB * numColsB * sizeof(float);
    size_t sizeC = numRowsA * numColsB * sizeof(float);

    float *matrixA = (float*)malloc(sizeA);
    float *matrixB = (float*)malloc(sizeB);
    float *matrixC = (float*)malloc(sizeC);

    fillMatrix(matrixA, numRowsA, numColsA);
    fillMatrix(matrixB, numRowsB, numColsB);

    matrixMultiplySharded(matrixA, matrixB, matrixC, numRowsA, numColsA, numColsB);

    free(matrixA);
    free(matrixB);
    free(matrixC);
    
    return 0;
}
