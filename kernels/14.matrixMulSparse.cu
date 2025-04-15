#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


/*

In the real-world, matrices can have a large number of zero elements. 
Operations on zero elements are redundant and waste computational resources. 
To avoid this, we can use sparse matrices to store only non-zero elements and perform operations on them.
One of such techniques is the Compressed Sparse Row (CSR) format. Sparse matrices are commonly used in the industry.

In the CSR format, we store the non-zero elements of the matrix in three arrays:

    1. Value (val): This array stores the value of each non-zero element.
    2. Column Index (colInd): This array stores the column index of each non-zero element.
    3. Row Pointer (rowPtr): This array stores the starting index of each row in the column index and value arrays.

For example, consider the following matrix:
    
    1 0 2
    0 0 3
    4 5 0

The CSR format of this matrix is:

    val = [1, 2, 3, 4, 5]
    colInd = [0, 2, 2, 0, 1]
    rowPtr = [0, 2, 3, 5]

To perform matrix multiplication on two sparse matrices A and B, we can use the following algorithm:
    
    1. For each row of matrix A:
        a. For each non-zero element in the row:
            i. Get the column index of the element.
            ii. For each non-zero element in the corresponding column of matrix B:
                A. If the column index of the element in matrix B matches the column index of the element in matrix A:
                    i. Multiply the values of the two elements and add the result to the sum.
        b. Store the sum in the result matrix C at the corresponding row and column index.


We will also henceforth start using an outer function to call the kernel function.
This is the standard practice in CUDA programming.

*/

__global__ void sparseMatrixMulKernel(
    const int *deviceRowPtrA, 
    const int *deviceColIndA, 
    const float *deviceValA,
    const int *deviceRowPtrB, 
    const int *deviceColIndB, 
    const float *deviceValB,
    float *deviceC, 
    int rowsA, 
    int colsB
    ) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA) {
        for (int col = 0; col < colsB; ++col) {
            float sum = 0.0f;

            for (int idxA = deviceRowPtrA[row]; idxA < deviceRowPtrA[row + 1]; ++idxA) {
                int colA = deviceColIndA[idxA];

                for (int idxB = deviceRowPtrB[colA]; idxB < deviceRowPtrB[colA + 1]; ++idxB) {
                    if (deviceColIndB[idxB] == col) {
                        sum += deviceValA[idxA] * deviceValB[idxB];
                    }
                }
            }
            deviceC[row * colsB + col] = sum;
        }
    }
}

void sparseMatrixMul(
    const std::vector<int>& hostRowPtrA, 
    const std::vector<int>& hostColIndA, 
    const std::vector<float>& hostValA,
    const std::vector<int>& hostRowPtrB, 
    const std::vector<int>& hostColIndB, 
    const std::vector<float>& hostValB,
    std::vector<float>& hostC, 
    int rowsA, 
    int colsB
    ) {

    int *deviceRowPtrA, *deviceColIndA, *deviceRowPtrB, *deviceColIndB;
    float *deviceValA, *deviceValB, *deviceC;
    int nnzA = hostValA.size();
    int nnzB = hostValB.size();
    size_t sizeC = rowsA * colsB * sizeof(float);

    cudaMalloc(&deviceRowPtrA, hostRowPtrA.size() * sizeof(int));
    cudaMalloc(&deviceColIndA, nnzA * sizeof(int));
    cudaMalloc(&deviceValA, nnzA * sizeof(float));
    cudaMalloc(&deviceRowPtrB, hostRowPtrB.size() * sizeof(int));
    cudaMalloc(&deviceColIndB, nnzB * sizeof(int));
    cudaMalloc(&deviceValB, nnzB * sizeof(float));
    cudaMalloc(&deviceC, sizeC);

    cudaMemcpy(deviceRowPtrA, hostRowPtrA.data(), hostRowPtrA.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceColIndA, hostColIndA.data(), nnzA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValA, hostValA.data(), nnzA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceRowPtrB, hostRowPtrB.data(), hostRowPtrB.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceColIndB, hostColIndB.data(), nnzB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceValB, hostValB.data(), nnzB * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(deviceC, 0, sizeC);

    int blockSize = 256;
    int numBlocks = (rowsA + blockSize - 1) / blockSize;

    sparseMatrixMulKernel<<<numBlocks, blockSize>>>(
        deviceRowPtrA, 
        deviceColIndA, 
        deviceValA,
        deviceRowPtrB, 
        deviceColIndB, 
        deviceValB,
        deviceC, 
        rowsA, 
        colsB);

    cudaMemcpy(hostC.data(), deviceC, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(deviceRowPtrA);
    cudaFree(deviceColIndA);
    cudaFree(deviceValA);
    cudaFree(deviceRowPtrB);
    cudaFree(deviceColIndB);
    cudaFree(deviceValB);
    cudaFree(deviceC);
}


int main() {

    int rowsA = 3;
    int colsB = 3;

    std::vector<int> hostRowPtrA = {0, 2, 4, 4};
    std::vector<int> hostColIndA = {0, 2, 1, 2};
    std::vector<float> hostValA = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<int> hostRowPtrB = {0, 2, 3, 4};
    std::vector<int> hostColIndB = {0, 1, 2, 0};
    std::vector<float> hostValB = {5.0f, 6.0f, 7.0f, 8.0f};

    std::vector<float> hostC(rowsA * colsB, 0.0f);

    sparseMatrixMul(
        hostRowPtrA, 
        hostColIndA, 
        hostValA, 
        hostRowPtrB, 
        hostColIndB, 
        hostValB, 
        hostC, 
        rowsA, 
        colsB);

    std::cout << "Result Matrix C:\n";
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            std::cout << hostC[i * colsB + j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
