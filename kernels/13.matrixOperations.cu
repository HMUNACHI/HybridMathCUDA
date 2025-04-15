#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void matrixTranspose(const float *a, float *c, int width, int height) {
    /*
    A transpose of a matrix is a new matrix whose rows are the columns of the original.
    For example:
    1 2 3
    4 5 6
    7 8 9

    Transpose:
    1 4 7
    2 5 8
    3 6 9

    Transpose a matrix by swapping the rows and columns: c[i][j] = a[j][i].
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        c[col * height + row] = a[row * width + col];
    }
}


__global__ void matrixTrace(const float *a, float *c, int width, int height) {
    /*
    The trace of a square matrix is the sum of the elements on the main diagonal.
    For example, the trace of the above matrix is 1 + 5 + 9 = 15:
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        if (row == col) {
            c[0] += a[row * width + col];
        }
    }
}

__global__ void matrixBroadcast(const float *a, float *c, int width, int height, float scalar) {
    /*
    Broadcasting is a technique used in matrix operations to apply an operation to all elements in a matrix.
    For example, to multiply all elements in a matrix by a scalar.
    */
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        c[row * width + col] = a[row * width + col] * scalar;
    }
}

void printMatrix(const float *matrix, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout << matrix[i * width + j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    // Matrixes are rather initialised without dynamic memory allocation.
    // This is to show CUDA can also feel natural.
    const int width = 3;
    const int height = 3;
    float hostA[width * height];
    float hostC[width * height] = {0};
    float hostTrace[1] = {0};
    float scalar = 2.0f;

    for (int i = 0; i < width * height; ++i) {
        hostA[i] = static_cast<float>(i + 1);
    }

    std::cout << "Original Matrix:\n";
    printMatrix(hostA, width, height);

    float *deviceA, *deviceC, *deviceTrace;
    size_t size = width * height * sizeof(float);

    cudaMalloc(&deviceA, size);
    cudaMalloc(&deviceC, size);
    cudaMalloc(&deviceTrace, sizeof(float));

    cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceTrace, hostTrace, sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    matrixTranspose<<<gridSize, blockSize>>>(deviceA, deviceC, width, height);
    cudaMemcpy(hostC, deviceC, size, cudaMemcpyDeviceToHost);
    std::cout << "\nTransposed Matrix:\n";
    printMatrix(hostC, height, width);

    matrixTrace<<<gridSize, blockSize>>>(deviceA, deviceTrace, width, height);
    cudaMemcpy(hostTrace, deviceTrace, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "\nTrace of the Matrix: " << hostTrace[0] << "\n";

    matrixBroadcast<<<gridSize, blockSize>>>(deviceA, deviceC, width, height, scalar);
    cudaMemcpy(hostC, deviceC, size, cudaMemcpyDeviceToHost);
    std::cout << "\nMatrix after broadcasting with scalar " << scalar << ":\n";
    printMatrix(hostC, width, height);

    cudaFree(deviceA);
    cudaFree(deviceC);
    cudaFree(deviceTrace);

    return 0;
}
