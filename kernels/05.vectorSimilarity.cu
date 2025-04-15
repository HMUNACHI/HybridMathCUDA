// Include necessary libraries
#include <ctime>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*

Cosine Similarity of Two Vectors: 
---------------------------------

Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space 
that measures the cosine of the angle between them. 

Its formula is given by: cos(θ) = (a . b) / (||a|| * ||b||)

where a and b are vectors and ||a|| and ||b|| are the magnitudes (norms) of a and b respectively.

Cosine similarity captures the orientation (or direction) of the vectors rather than their magnitude. 
If two vectors point in the same direction, the cosine of the angle between them is 1. 
If they are orthogonal (at 90 degrees to each other), the cosine of the angle is 0, indicating no similarity. 
If they point in completely opposite directions, the cosine of the angle is -1.

Mathematically, cosine similarity is the dot product of the two vectors divided by the product of their magnitudes. 
This normalization makes cosine similarity independent of the vector magnitudes, focusing solely on the direction.

n natural language processing (NLP), documents and text are often represented as vectors in a high-dimensional space. 
Cosine similarity is widely used to compare these document vectors because it effectively measures how similar the documents are 
in terms of their content, independent of their length.

This CUDA implementation demonstrates how to use multiple kernels sequentially to parallelise the calculation of cosine similarity.
The main code will from this point be rid of comments for clarity and brevity but wityh readale code.

*/

__global__ void dotProduct(const float *a, const float *b, float *c, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(c, a[idx] * b[idx]);
    }
}

__global__ void vectorMagnitude(const float *vector, float *magnitude, int numElements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements){
        atomicAdd(magnitude, vector[idx] * vector[idx]);
    }
}

float randomFloat(int randMax = 1000){
    return static_cast<float>(rand()) / static_cast<float>(randMax);
}

int main(){
    srand(time(0));
    int numElements = 300;
    size_t size = numElements * sizeof(float);
    float *hostVectorA, *hostVectorB, *hostDotProduct, *hostMagnitudeA, *hostMagnitudeB;
    float *deviceVectorA, *deviceVectorB, *deviceDotProduct, *deviceMagnitudeA, *deviceMagnitudeB;

    hostVectorA = (float*) malloc(size);
    hostVectorB = (float*) malloc(size);
    hostDotProduct = (float*) malloc(sizeof(float));
    hostMagnitudeA = (float*) malloc(sizeof(float));
    hostMagnitudeB = (float*) malloc(sizeof(float));

    for (int idx = 0; idx < numElements; idx++){
        hostVectorA[idx] = randomFloat();
        hostVectorB[idx] = randomFloat();
    }

    cudaMalloc((void**)&deviceVectorA, size);
    cudaMalloc((void**)&deviceVectorB, size);
    cudaMalloc((void**)&deviceDotProduct, sizeof(float));
    cudaMalloc((void**)&deviceMagnitudeA, sizeof(float));
    cudaMalloc((void**)&deviceMagnitudeB, sizeof(float));

    cudaMemcpy(deviceVectorA, hostVectorA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVectorB, hostVectorB, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    dotProduct<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorA, deviceVectorB, deviceDotProduct, numElements);
    vectorMagnitude<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorA, deviceMagnitudeA, numElements);
    vectorMagnitude<<<blocksPerGrid, threadsPerBlock>>>(deviceVectorB, deviceMagnitudeB, numElements);

    cudaMemcpy(hostDotProduct, deviceDotProduct, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostMagnitudeA, deviceMagnitudeA, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostMagnitudeB, deviceMagnitudeB, sizeof(float), cudaMemcpyDeviceToHost);

    *hostMagnitudeA = sqrt(*hostMagnitudeA);
    *hostMagnitudeB = sqrt(*hostMagnitudeB);

    float cosineSimilarity = *hostDotProduct / (*hostMagnitudeA * *hostMagnitudeB);

    std::cout << "Cosine Similarity: " << cosineSimilarity << std::endl;

    free(hostVectorA);
    free(hostVectorB);
    free(hostDotProduct);
    free(hostMagnitudeA);
    free(hostMagnitudeB);

    cudaFree(deviceVectorA);
    cudaFree(deviceVectorB);
    cudaFree(deviceDotProduct);
    cudaFree(deviceMagnitudeA);
    cudaFree(deviceMagnitudeB);
}