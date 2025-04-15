// Include necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*

Correlation Coefficient of Two Vectors:
----------------------------------------    
Vector correlation is a measure of the relationship between two or more variables.
Where -1 indicates a perfect negative correlation, 0 indicates no correlation, and 1 indicates a perfect positive correlation.
The formula for the correlation coefficient is given by: 

Pearson r = Σ((xi - x̄)(yi - ȳ)) / √(Σ(xi - x̄)² * Σ(yi - ȳ)²)

Cosine similarity focuses on the angle between vectors, measuring how similar their directions are regardless of magnitude.
While correlation takes into account the mean and standard deviation of the variables; how the variables change together.
Not their angle of inclination when plotted in a multi-dimensional space.

This CUDA demonstrates how to breakdown complex equations into multiple sub-kernels and yet parallelise.
However, launching kernels takes overhead cost, so it's not always beneficial to break down into multiple kernels.

*/

__global__ void calculateNumerator(const float *a, const float *b, float *numerator, float meanA, float meanB, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(numerator, (a[idx] - meanA) * (b[idx] - meanB));
    }
}

__global__ void calculateDenominator(const float *vector, float *denominator, float meanB, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(denominator, (vector[idx] - meanB) * (vector[idx] - meanB));
    }
}

float randomFloat(int randMax = 1000){
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

float calculateMean(float* vector, int numElements){
    float sum = 0;
    for(int i = 0; i < numElements; i++){
        sum += vector[i];
    }
    return sum / numElements;
}

int main(){
    srand(time(0));
    int numElements = 300;
    size_t size = numElements * sizeof(float);
    float *hostVectorA, *hostVectorB, *hostNumerator, *hostDenominatorA, *hostDenominatorB;
    float *deviceVectorA, *deviceVectorB, *deviceNumerator, *deviceDenominatorA, *deviceDenominatorB;

    hostVectorA = (float*) malloc(size);
    hostVectorB = (float*) malloc(size);
    hostNumerator = (float*) malloc(sizeof(float));
    hostDenominatorA = (float*) malloc(sizeof(float));
    hostDenominatorB = (float*) malloc(sizeof(float));

    for (int idx = 0; idx < numElements; idx++){
        hostVectorA[idx] = randomFloat();
        hostVectorB[idx] = randomFloat();
    }

    float meanA = calculateMean(hostVectorA, numElements);
    float meanB = calculateMean(hostVectorB, numElements);

    cudaMalloc((void**)&deviceVectorA, size);
    cudaMalloc((void**)&deviceVectorB, size);
    cudaMalloc((void**)&deviceNumerator, sizeof(float));
    cudaMalloc((void**)&deviceDenominatorA, sizeof(float));
    cudaMalloc((void**)&deviceDenominatorB, sizeof(float));

    cudaMemcpy(deviceVectorA, hostVectorA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVectorB, hostVectorB, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    calculateNumerator<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorA, deviceVectorB, deviceNumerator, meanA, meanB, numElements);
    calculateDenominator<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorA, deviceDenominatorA, meanA, numElements);
    calculateDenominator<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorB, deviceDenominatorB, meanB, numElements);

    cudaMemcpy(hostNumerator, deviceNumerator, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostDenominatorA, deviceDenominatorA, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostDenominatorB, deviceDenominatorB, sizeof(float), cudaMemcpyDeviceToHost);

    float correlationCoefficient = *hostNumerator / sqrt(*hostDenominatorA * *hostDenominatorB);

    std::cout << "Correlation Coefficient: " << correlationCoefficient << std::endl;

    free(hostVectorA);
    free(hostVectorB);
    free(hostNumerator);
    free(hostDenominatorA);
    free(hostDenominatorB);

    cudaFree(deviceVectorA);
    cudaFree(deviceVectorB);
    cudaFree(deviceNumerator);
    cudaFree(deviceDenominatorA);
    cudaFree(deviceDenominatorB);
}