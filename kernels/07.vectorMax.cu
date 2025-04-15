// Include necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* 

CUDA's in-built atomicMax only works for ints, hence we use atomicCAS, stands for Atomic Compare And Swap. 
The function takes a memory address, a comparison value, and a new value as arguments. 
It compares the value at the given memory address with the comparison value. 
If they match, it swaps the value at the memory address with the new value. 
The function returns the original value that was at the memory address before the operation. 
This operation is atomic, meaning it's guaranteed to be performed without interruption by other threads. 
This is crucial when many threads may be trying to update the same memory location simultaneously.

*/

__device__ float atomicMax(float* address, float val)

{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void vectorMax(const float *vector, float *max, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicMax(max, vector[idx]);
    }
}


int main(){
    srand(time(0));
    int numElements = 1000000;
    size_t size = numElements * sizeof(float);
    float *hostVector, *deviceVector;
    float *hostMax, *deviceMax;

    hostVector = (float*) malloc(size);
    hostMax = (float*) malloc(sizeof(float));

    // Initialise a random vector of distribution
    for (int idx = 0; idx <= numElements; idx++){
        hostVector[idx] = 1.0f * idx;
    }

    // Allocate device vector and variables
    cudaMalloc((void**)&deviceVector, size);
    cudaMalloc((void**)&deviceMax, sizeof(float));

    // Copy host vector to device
    cudaMemcpy(deviceVector, hostVector, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector max kernel
    vectorMax<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, deviceMax, numElements);
    cudaDeviceSynchronize();

    // Copy the result from device to host and print
    cudaMemcpy(hostMax, deviceMax, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Max: %f\n", *hostMax);

    // Free the memory allocated for host 
    free(hostVector);
    free(hostMax);

    // Free the memory allocated for device 
    cudaFree(deviceVector);
    cudaFree(deviceMax);
}