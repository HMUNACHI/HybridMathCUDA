#include <iostream> // for std::cout
#include <cuda_runtime.h> // for CUDA runtime functions
#include <device_launch_parameters.h> // for CUDA kernel launch parameters

/* 

Addition of Two Vectors: 
--------------------------
a + b = (a1 + b1, a2 + b2, ..., an + bn)

*/
__global__ void vectorAdd(const int *a, 
                          const int *b, 
                          int *c, 
                          int numElements) {
    // Calculate the global index for the current thread
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (i < numElements){
        // Add corresponding elements of input vectors
        c[i] = a[i] + b[i];
    }
}

int main() {
    // Number of elements in vectors
    int numElements = 50;

    // Calculate the size of the data
    size_t size = numElements * sizeof(int);

    // Declare pointers for host and device vectors
    int *hostA, *hostB, *hostC;
    int *deviceA, *deviceB, *deviceC;

    // Allocate memory for host vectors
    hostA = (int*) malloc(size);
    hostB = (int*) malloc(size);
    hostC = (int*) malloc(size);

    // Initialize input vectors
    for (int i = 0; i < numElements; i++){
        hostA[i] = i;
        hostB[i] = i*2;
    }

    // Allocate memory for device vectors
    cudaMalloc((void**)&deviceA, size);
    cudaMalloc((void**)&deviceB, size);
    cudaMalloc((void**)&deviceC, size);

    // Copy host vectors to device
    cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vectorAdd kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, numElements);

    // Copy result vector from device to host
    cudaMemcpy(hostC, deviceC, size, cudaMemcpyDeviceToHost);

    // Print the results
    for (int i = 0; i < numElements; i++){
        std::cout << hostA[i] << " + " << hostB[i] << " = " << hostC[i] << std::endl;
    }

    // Free the memory allocated for host vectors
    free(hostA);
    free(hostB);
    free(hostC);

    // Free the memory allocated for device vectors
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    return 0;
}