// Include necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* 
Dot Product of Two Vectors: 
--------------------------
Given two vectors 'a' and 'b' of the same dimension 'n', their dot product is defined as: 

a · b = a1 * b1 + a2 * b2 + ... + an * bn

Geometrically, the dot product measures the cosine of the angle between the vectors scaled by their magnitudes.  
It's useful for tasks like determining if vectors are orthogonal (dot product is zero), projecting one vector onto another, calculating work done by a force.
Dot products are commutative, distributive, and linear. 


CUDA Tweaks:
--------------------
In CUDA, we can exploit parallelism to accelerate the calculation of dot products, especially for large vectors. 
However, naive parallelization can lead to race conditions when multiple threads attempt to update a shared sum simultaneously. 
So we at such structure our code in the following way:

1. Thread Assignment: Each thread is assigned to compute the product of one pair of elements from the vectors (e.g., thread 1 computes a1 * b1).
2. Partial Products: Each thread stores its partial product in a local variable.
3. Atomic Addition: Instead of directly adding their partial products to a shared sum, each thread uses atomicAdd() to perform an atomic update. This guarantees that only one thread updates the shared sum at a time, avoiding race conditions. 
4. Synchronization (Optional): A synchronization barrier (e.g., __syncthreads()) may be needed to ensure all threads have completed their computations before the final sum is used.

*/

// CUDA kernel function to compute the dot product of two vectors
__global__ void vectorDot(const float *a, const float *b, float *c, int numElements) {
    // Calculate the unique index for the thread
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // Check if index is within the range of elements
    if (idx < numElements){
        // Perform the multiplication and add to the result
        atomicAdd(c, a[idx] * b[idx]);
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
    int numElements = 300;

    // Calculate the size of the vectors in bytes
    size_t size = numElements * sizeof(float);
    
    // Declare pointers for host and device vectors
    float *hostA, *hostB, *hostC;
    float *deviceA, *deviceB, *deviceC;

    // Allocate memory for host vectors
    hostA = (float*) malloc(size);
    hostB = (float*) malloc(size);
    hostC = (float*) malloc(sizeof(float)); // Only need space for one float for the result

    // Initialize host vectors with random floats
    for (int idx = 0; idx < numElements; idx++){
        hostA[idx] = randomFloat();
        hostB[idx] = randomFloat();
    }

    // Allocate memory for device vectors
    cudaMalloc((void**)&deviceA, size);
    cudaMalloc((void**)&deviceB, size);
    cudaMalloc((void**)&deviceC, sizeof(float)); // Only need space for one float for the result

    // Copy host vectors to device
    cudaMemcpy(deviceA, hostA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector dot product kernel
    vectorDot<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, numElements);

    // Copy the result from device to host
    cudaMemcpy(hostC, deviceC, sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result of the dot product
    std::cout << "Dot product: " << *hostC << std::endl;

    // Free the memory allocated for host vectors
    free(hostA);
    free(hostB);
    free(hostC);

    // Free the memory allocated for device vectors
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
}