// Include necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/* 

The math:
---------------
The Gaussian function, also known as the normal distribution, is a bell-shaped curve that describes many natural phenomena. 
The formula for a Gaussian function is: f(x) = max * exp(-((x - mean)^2) / (2 * std^2)).
This is not to be confused with (1 / √(2π)) * exp(-x^2 / 2), a special case of the Gaussian function 
where the mean is 0 and the standard deviation is 1.

CUDA implementation:
--------------------
This CUDA program uses multiple global kernels to calculate the mean, standard deviation, maximum value, 
and then the Gaussian function whioch models a vector of distributions as a gaussian.
This demos solving real-word problems with maths and accelating the computation with GPU.
We resuse many concepts from the preceding files so they sink in.

*/

__global__ void vectorMean(const float *vector, float *mean, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(mean, vector[idx]);
    }
}

__global__ void vectorStd(const float *vector, float *std, float *mean, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(std, pow(vector[idx] - *mean, 2));
    }
}

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

__global__ void vectorGaussian(const float *vector, 
                          float *gaussian, 
                          int numElements, 
                          float mean, 
                          float std, 
                          float max){

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        gaussian[idx] = max * exp(-pow(vector[idx] - mean, 2) / (2 * pow(std, 2)));
    }
}

float randomFloat(int randMax = 1000){
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

int main(){
    srand(time(0));
    int numElements = 300;
    size_t size = numElements * sizeof(float);
    float *hostVector, *deviceVector;
    float *hostMean, *deviceMean;
    float *hostStd, *deviceStd;
    float *hostMax, *deviceMax;
    float *hostGaussian, *deviceGaussian;

    hostVector = (float*) malloc(size);
    hostMean = (float*) malloc(sizeof(float));
    hostStd = (float*) malloc(sizeof(float));
    hostMax = (float*) malloc(sizeof(float));
    hostGaussian = (float*) malloc(size);

    // Initialise a random vector of distribution
    for (int idx = 0; idx < numElements; idx++){
        hostVector[idx] = randomFloat();
    }

    // Allocate device vector and variables
    cudaMalloc((void**)&deviceVector, size);
    cudaMalloc((void**)&deviceMean, sizeof(float));
    cudaMalloc((void**)&deviceStd, sizeof(float));
    cudaMalloc((void**)&deviceMax, sizeof(float));
    cudaMalloc((void**)&deviceGaussian, size);

    // Copy host vectors to device
    cudaMemcpy(deviceVector, hostVector, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks per grid
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the vector mean kernel
    vectorMean<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, deviceMean, numElements);
    cudaDeviceSynchronize();
    cudaMemcpy(hostMean, deviceMean, sizeof(float), cudaMemcpyDeviceToHost);
    *hostMean /= numElements;

    // Launch the vector std kernel
    vectorStd<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, deviceStd, deviceMean, numElements);
    cudaDeviceSynchronize();
    cudaMemcpy(hostStd, deviceStd, sizeof(float), cudaMemcpyDeviceToHost);
    *hostStd = sqrt(*hostStd / numElements);

    // Launch the vector max kernel
    vectorMax<<<blocksPerGrid, threadsPerBlock>>>(deviceVector, deviceMax, numElements);
    cudaDeviceSynchronize();
    cudaMemcpy(hostMax, deviceMax, sizeof(float), cudaMemcpyDeviceToHost);

    // Launch the vector gaussian kernel
    vectorGaussian<<<blocksPerGrid, threadsPerBlock>>>(
        deviceVector, 
        deviceGaussian, 
        numElements, 
        *hostMean, 
        *hostStd, 
        *hostMax
        );

    // Copy the result from device to host
    cudaMemcpy(hostGaussian, deviceGaussian, size, cudaMemcpyDeviceToHost);

    // Print the results of the Gaussian function
    for (int idx = 0; idx < numElements; idx++){
        std::cout << "Gaussian function value at x = " << hostVector[idx] << ": " << hostGaussian[idx] << std::endl;
    }

    // Free the memory allocated for host vectors
    free(hostVector);
    free(hostMean);
    free(hostStd);
    free(hostMax);
    free(hostGaussian);

    // Free the memory allocated for device vectors
    cudaFree(deviceVector);
    cudaFree(deviceMean);
    cudaFree(deviceStd);
    cudaFree(deviceMax);
    cudaFree(deviceGaussian);
}
