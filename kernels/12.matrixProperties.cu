#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Matrix operations in CUDA are often achieved with index manipulation, these kernels demonstrates these.

__global__ void isToeplitz(const float *a, int width, int height, bool *result) {
    /*
    A matrix is said to be Toeplitz if every diagonal from top-left to bottom-right has the same element.
    For example, the following matrix is a Toeplitz matrix:
    1 2 3 4
    5 1 2 3
    6 5 1 2

    The following matrix is not a Toeplitz matrix:
    1 2 3 4
    5 1 2 3
    6 7 1 2

    The element at (i, j) is equal to the element at (i - 1, j - 1) for all i, j.
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        if (row > 0 && col > 0) {
            if (a[row * width + col] != a[(row - 1) * width + (col - 1)]) {
                *result = false;
            }
        }
    }
}


__global__ void isDiagonal(const float *a, int width, int height, bool *result) {
    /*
    A matrix is said to be diagonal if all the elements outside the main diagonal are zero.
    For example, the following matrix is a diagonal matrix:
    1 0 0 0
    0 2 0 0
    0 0 3 0
    0 0 0 4

    The following matrix is not a diagonal matrix:
    1 0 0 0
    0 2 0 0
    0 0 3 0
    0 0 0 5

    The element at (i, j) is zero for all i != j.
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        if (row != col) {
            if (a[row * width + col] != 0) {
                *result = false;
            }
        }
    }
}


__global__ void isSymmetric(const float *a, int width, int height, bool *result) {
    /*
    A matrix is said to be symmetric if it is equal to its transpose.
    For example, the following matrix is a symmetric matrix:
    1 2 3
    2 4 5
    3 5 6

    The following matrix is not a symmetric matrix:
    1 2 3
    4 5 6
    7 8 9

    The element at (i, j) is equal to the element at (j, i) for all i, j.
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        if (a[row * width + col] != a[col * width + row]) {
            *result = false;
        }
    }
}


__global__ void isSkewSymmetric(const float *a, int width, int height, bool *result) {
    /*
    A matrix is said to be skew symmetric if it is equal to the negative of its transpose.
    For example, the following matrix is a skew symmetric matrix:
    0 2 -5
    -2 0 -3
    5 3 0

    The following matrix is not a skew symmetric matrix:
    0 2 -5
    2 0 -3
    5 3 0

    The element at (i, j) is equal to the negative of the element at (j, i) for all i, j.
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        if (a[row * width + col] != -a[col * width + row]) {
            *result = false;
        }
    }
}


__global__ void isIdempotent(const float *a, int width, int height, bool *result) {
    /*
    A matrix is said to be idempotent if it is equal to its square.
    For example, the following matrix is an idempotent matrix:
    1 2
    3 4

    The following matrix is not an idempotent matrix:
    1 2
    3 5

    The element at (i, j) is equal to the element at (i, j) squared for all i, j.
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        if (a[row * width + col] != a[row * width + col] * a[row * width + col]) {
            *result = false;
        }
    }
}


int main() {

    int width = 4;
    int height = 4;

    float *hostMatrix = (float*)malloc(width * height * sizeof(float));
    bool *hostToeplitzResult = (bool*)malloc(sizeof(bool));
    bool *hostDiagonalResult = (bool*)malloc(sizeof(bool));
    bool *hostSymmetricResult = (bool*)malloc(sizeof(bool));
    bool *hostSkewSymmetricResult = (bool*)malloc(sizeof(bool));
    bool *hostIdempotentResult = (bool*)malloc(sizeof(bool));

    float *deviceMatrix;
    bool *deviceToeplitzResult;
    bool *deviceDiagonalResult;
    bool *deviceSymmetricResult;
    bool *deviceSkewSymmetricResult;
    bool *deviceIdempotentResult;

    cudaMalloc((void**)&deviceMatrix, width * height * sizeof(float));
    cudaMalloc((void**)&deviceToeplitzResult, sizeof(bool));
    cudaMalloc((void**)&deviceDiagonalResult, sizeof(bool));
    cudaMalloc((void**)&deviceSymmetricResult, sizeof(bool));
    cudaMalloc((void**)&deviceSkewSymmetricResult, sizeof(bool));
    cudaMalloc((void**)&deviceIdempotentResult, sizeof(bool));

    srand(time(NULL));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            hostMatrix[i * width + j] = rand() % 10;
        }
    }

    *hostToeplitzResult = true;
    *hostDiagonalResult = true;
    *hostSymmetricResult = true;
    *hostSkewSymmetricResult = true;
    *hostIdempotentResult = true;
    
    cudaMemcpy(deviceMatrix, hostMatrix, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceToeplitzResult, hostToeplitzResult, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceDiagonalResult, hostDiagonalResult, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSymmetricResult, hostSymmetricResult, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceSkewSymmetricResult, hostSkewSymmetricResult, sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceIdempotentResult, hostIdempotentResult, sizeof(bool), cudaMemcpyHostToDevice);

    dim3 blockSize(2, 2);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    isToeplitz<<<gridSize, blockSize>>>(deviceMatrix, width, height, deviceToeplitzResult);
    isDiagonal<<<gridSize, blockSize>>>(deviceMatrix, width, height, deviceDiagonalResult);
    isSymmetric<<<gridSize, blockSize>>>(deviceMatrix, width, height, deviceSymmetricResult);
    isSkewSymmetric<<<gridSize, blockSize>>>(deviceMatrix, width, height, deviceSkewSymmetricResult);
    isIdempotent<<<gridSize, blockSize>>>(deviceMatrix, width, height, deviceIdempotentResult);

    cudaMemcpy(hostToeplitzResult, deviceToeplitzResult, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostDiagonalResult, deviceDiagonalResult, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSymmetricResult, deviceSymmetricResult, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostSkewSymmetricResult, deviceSkewSymmetricResult, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostIdempotentResult, deviceIdempotentResult, sizeof(bool), cudaMemcpyDeviceToHost);

    if (*hostToeplitzResult) {
        std::cout << "Matrix is Toeplitz" << std::endl;
    } else {
        std::cout << "Matrix is not Toeplitz" << std::endl;
    }

    if (*hostDiagonalResult) {
        std::cout << "Matrix is Diagonal" << std::endl;
    } else {
        std::cout << "Matrix is not Diagonal" << std::endl;
    }

    if (*hostSymmetricResult) {
        std::cout << "Matrix is Symmetric" << std::endl;
    } else {
        std::cout << "Matrix is not Symmetric" << std::endl;
    }

    if (*hostSkewSymmetricResult) {
        std::cout << "Matrix is Skew Symmetric" << std::endl;
    } else {
        std::cout << "Matrix is not Skew Symmetric" << std::endl;
    }

    if (*hostIdempotentResult) {
        std::cout << "Matrix is Idempotent" << std::endl;
    } else {
        std::cout << "Matrix is not Idempotent" << std::endl;
    }

    free(hostMatrix);
    free(hostToeplitzResult);
    free(hostDiagonalResult);
    free(hostSymmetricResult);
    free(hostSkewSymmetricResult);
    free(hostIdempotentResult);

    cudaFree(deviceMatrix);
    cudaFree(deviceToeplitzResult);
    cudaFree(deviceDiagonalResult);
    cudaFree(deviceSymmetricResult);
    cudaFree(deviceSkewSymmetricResult);
    cudaFree(deviceIdempotentResult);

    return 0;
    
}