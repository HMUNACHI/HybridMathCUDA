// Include necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel for element-wise multiplication of two vectors aka Hadamard product
__global__ void vectorMul(const float *a,
                          const float *b,
                          float *c,
                          float numElements){
    // Calculate the unique index for the thread
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Check if index is within the range of elements
    if (idx < numElements){
        // Perform the multiplication
        c[idx] = a[idx] * b[idx];
    }
}

// Function to generate a random float
float randomFloat(int randMax = 1000){
    // Return a random float between 0 and 1
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

// Main function
int main(){
    // Seed the random number generator
    srand(time(0));

    // Define the number of elements in the vectors
    int numElements = 100;

    // Calculate the size of the vectors in bytes
    size_t size = numElements * sizeof(float);
    
    // Declare pointers for host and device vectors
    float *hostA, *hostB, *hostC;
    float *deviceA, *deviceB, *deviceC;

    // Allocate memory for host vectors
    hostA = (float*) malloc(size);
    hostB = (float*) malloc(size);
    hostC = (float*) malloc(size);

    // Initialize host vectors with random floats
    for (int idx = 0; idx < numElements; idx++){
        hostA[idx] = randomFloat();
        hostB[idx] = randomFloat();
    }

    // Allocate memory for device vectors
    cudaMalloc((void**)&deviceA, size);
    cudaMalloc((void**)&deviceB, size);
    cudaMalloc((void**)&deviceC, size);

    // Copy host vectors to device
    cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector multiplication kernel
    vectorMul<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, numElements);

    // Copy the result vector from device to host
    cudaMemcpy(hostC, deviceC, size, cudaMemcpyDeviceToHost);

    // Print the results of the multiplication
    for (int idx = 0; idx < numElements; idx++){
        std::cout << hostA[idx] << " * " << hostB[idx] << " = " << hostC[idx] << std::endl;
    }

    // Free the memory allocated for host vectors
    free(hostA);
    free(hostB);
    free(hostC);

    // Free the memory allocated for device vectors
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}
