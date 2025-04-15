#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* 
Matrix Multiplication: 
--------------------------
Given two matrices 'A' and 'B' of dimensions 'm x n' and 'n x p' respectively, 
their product is a new matrix 'C' of dimension 'm x p' where each element is defined as: 

Cij = Ai1 * B1j + Ai2 * B2j + ... + Ain * Bnj

Geometrically, matrix multiplication can be seen as a linear transformation applied to the vectors in the space.  
It's useful for tasks like transforming coordinates, solving systems of linear equations, 
and performing rotations and scaling in computer graphics.
Matrix multiplication is associative and distributive, but not commutative. 


CUDA Tweaks:
--------------------
1. Thread Assignment: Each thread is assigned to compute one element of the resulting matrix. 
   The index for each thread is calculated based on the block size, block index, and thread index within the block 
   (e.g., thread (i, j) computes Cij).

2. Element Calculation: Each thread calculates the corresponding element in the resulting matrix 
   by multiplying the elements in the corresponding row of the first matrix and the corresponding column 
   of the second matrix, and summing up these products.

The use of atomic operations is not required in this case as each thread is responsible for calculating
and storing a unique element in the resulting matrix, avoiding race conditions.
It is tempting to consider using launching vectorDot kernels for each element of the resulting matrix,
This is not recommended as it would lead to a large number of kernel launches and resource overheads. 

*/

__global__ void matrixMul(const float *a, const float *b, float *c, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        float value = 0;
        for (int idx = 0; idx < width; ++idx) {
            value += a[row * width + idx] * b[idx * width + col];
        }
        c[row * width + col] = value;
    }
}

float randomFloat(int randMax = 1000){
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

int main(){
    srand(time(0));

    int width = 6;
    int height = 4;
    size_t size = width * height * sizeof(float);
    float *hostA, *hostB, *hostC;
    float *deviceA, *deviceB, *deviceC;

    hostA = (float*) malloc(size);
    hostB = (float*) malloc(size);
    hostC = (float*) malloc(size);

    for (int i = 0; i < width * height; i++){
        hostA[i] = randomFloat();
        hostB[i] = randomFloat();
    }

    cudaMalloc((void**)&deviceA, size);
    cudaMalloc((void**)&deviceB, size);
    cudaMalloc((void**)&deviceC, size);

    cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x, (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, width, height);
    cudaMemcpy(hostC, deviceC, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            std::cout << hostC[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    free(hostA);
    free(hostB);
    free(hostC);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}