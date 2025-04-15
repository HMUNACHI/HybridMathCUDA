// Include necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* 

The math:
---------------
In scalars, we know that 10 > 5 and 6 < 100 but how do we similarly quantify vectors?
We take their norms, and L2 norm is given by:

||v||₂ = √(v₁² + v₂² + ... + vₙ²)

In reality, this equivalent to quantify the distance from the origin to the point defined by the vector.
But for norms we assume the origin of all axis to be 0, this can be expanded to the folllowing:

||v||₂ = √((v₁-0)² + (v₂-0)² + ... + (vₙ-0)²)


CUDA implementation:
--------------------
This CUDA program implements the generic Lp Norm of a vector:

||v||ₚ = (|v₁|ᵖ + |v₂|ᵖ + ... + |vₙ|ᵖ)^(1/p)

We again use atomic addition to aggregate the results of the square of each element, then take the sqrt.

*/

__global__ void vectorNorm(const float *vector, float *norm, int p, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        // Perform the multiplication and add to the result
        atomicAdd(norm, pow(vector[idx], p));
    }
}

float randomFloat(int randMax = 1000){
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

int main(){
    // Initialize pointer for host and device vectors
    srand(time(0));
    int p = 2;
    int numElements = 300;
    size_t size = numElements * sizeof(float);
    float *hostVector, *hostNorm;
    float *deviceVector, *devicetNorm;

    // Allocate memory for host vectors
    hostVector = (float*) malloc(size);
    hostNorm = (float*) malloc(sizeof(float)); // Only need space for one float for the result

    // Initialize host vectors with random floats
    for (int idx = 0; idx < numElements; idx++){
        hostVector[idx] = randomFloat();
    }

    // Allocate memory for device vectors
    cudaMalloc((void**)&deviceVector, size);
    cudaMalloc((void**)&devicetNorm, sizeof(float)); // Only need space for one float for the result

    // Copy host vectors to device
    cudaMemcpy(deviceVector, hostVector, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector vector norm kernel
    vectorNorm<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, devicetNorm, p, numElements);

    // Copy the result from device to host
    cudaMemcpy(hostNorm, devicetNorm, sizeof(float), cudaMemcpyDeviceToHost);

    // take the square root of the result
    *hostNorm = pow(*hostNorm, 1.0/p);

    // Print the result of the vector norm
    std::cout << "Vector Norm: " << *hostNorm << std::endl;

    // Free the memory allocated for host vectors
    free(hostVector);
    free(hostNorm);

    // Free the memory allocated for device vectors
    cudaFree(deviceVector);
    cudaFree(devicetNorm);
}
