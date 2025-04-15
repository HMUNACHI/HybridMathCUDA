#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

/*
Efficient convolution implementation using CUDA.

Convolution is a fundamental operation in deep learning used in Convolutional Neural Networks (CNNs).
The operation slides a small kernel (filter) over an input (typically an image) to produce an output feature map.

The naive approach would be to compute each output pixel with a separate thread, where each thread
performs all the multiplications and additions for one output pixel. However, this leads to:
1. Redundant memory accesses (adjacent threads load overlapping input data)
2. Inefficient use of memory bandwidth

Optimizations implemented:
1. Shared memory tiling: Load blocks of input data into shared memory to reduce global memory accesses
2. Register usage: Store frequently accessed data in registers
3. Memory coalescing: Ensure threads access contiguous memory locations
4. Loop unrolling: Reduce loop overhead for small, fixed-size kernels

These optimizations significantly improve convolution performance compared to naive implementations.
*/

const int TILE_SIZE = 16;  // Tile size for shared memory optimization
const int KERNEL_SIZE = 3; // Fixed kernel size (3x3)

// Efficient convolution kernel using shared memory
__global__ void efficientConvolution2D(
    const float* input, 
    const float* kernel, 
    float* output, 
    int inputWidth, 
    int inputHeight,
    int outputWidth, 
    int outputHeight
) {
    // Calculate output coordinates
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Shared memory for input tile
    __shared__ float sharedInput[TILE_SIZE + KERNEL_SIZE - 1][TILE_SIZE + KERNEL_SIZE - 1];
    
    // Calculate input coordinates for this tile
    int inX = outX - KERNEL_SIZE/2;
    int inY = outY - KERNEL_SIZE/2;
    
    // Load input tile into shared memory
    int sharedX = threadIdx.x;
    int sharedY = threadIdx.y;
    
    // Each thread loads one element into shared memory
    // Main tile
    if (sharedX < TILE_SIZE && sharedY < TILE_SIZE) {
        // Boundary check
        if (inX >= 0 && inX < inputWidth && inY >= 0 && inY < inputHeight) {
            sharedInput[sharedY][sharedX] = input[inY * inputWidth + inX];
        } else {
            sharedInput[sharedY][sharedX] = 0.0f;  // Zero padding
        }
    }
    
    // Load halo regions (additional threads load the border elements)
    if (sharedX < KERNEL_SIZE - 1 && sharedY < TILE_SIZE) {
        // Right halo
        int haloX = inX + TILE_SIZE;
        if (haloX < inputWidth && inY >= 0 && inY < inputHeight) {
            sharedInput[sharedY][sharedX + TILE_SIZE] = input[inY * inputWidth + haloX];
        } else {
            sharedInput[sharedY][sharedX + TILE_SIZE] = 0.0f;  // Zero padding
        }
    }
    
    if (sharedX < TILE_SIZE && sharedY < KERNEL_SIZE - 1) {
        // Bottom halo
        int haloY = inY + TILE_SIZE;
        if (inX >= 0 && inX < inputWidth && haloY < inputHeight) {
            sharedInput[sharedY + TILE_SIZE][sharedX] = input[haloY * inputWidth + inX];
        } else {
            sharedInput[sharedY + TILE_SIZE][sharedX] = 0.0f;  // Zero padding
        }
    }
    
    if (sharedX < KERNEL_SIZE - 1 && sharedY < KERNEL_SIZE - 1) {
        // Bottom-right corner halo
        int haloX = inX + TILE_SIZE;
        int haloY = inY + TILE_SIZE;
        if (haloX < inputWidth && haloY < inputHeight) {
            sharedInput[sharedY + TILE_SIZE][sharedX + TILE_SIZE] = input[haloY * inputWidth + haloX];
        } else {
            sharedInput[sharedY + TILE_SIZE][sharedX + TILE_SIZE] = 0.0f;  // Zero padding
        }
    }
    
    // Ensure all threads have loaded the shared memory
    __syncthreads();
    
    // Compute convolution if within output bounds
    if (outX < outputWidth && outY < outputHeight) {
        float sum = 0.0f;
        
        // Unrolled loop for 3x3 kernel for better performance
        #pragma unroll
        for (int ky = 0; ky < KERNEL_SIZE; ky++) {
            #pragma unroll
            for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                sum += sharedInput[threadIdx.y + ky][threadIdx.x + kx] * 
                       kernel[ky * KERNEL_SIZE + kx];
            }
        }
        
        // Write result to output
        output[outY * outputWidth + outX] = sum;
    }
}

// Helper function to fill matrix with random values
void fillMatrix(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

int main() {
    // Define input dimensions
    int inputWidth = 1024;
    int inputHeight = 1024;
    int kernelSize = KERNEL_SIZE;
    
    // Calculate output dimensions
    int outputWidth = inputWidth - kernelSize + 1;
    int outputHeight = inputHeight - kernelSize + 1;
    
    // Allocate host memory
    std::vector<float> hostInput(inputWidth * inputHeight);
    std::vector<float> hostKernel(kernelSize * kernelSize);
    std::vector<float> hostOutput(outputWidth * outputHeight);
    
    // Fill input and kernel with random data
    fillMatrix(hostInput.data(), inputWidth * inputHeight);
    fillMatrix(hostKernel.data(), kernelSize * kernelSize);
    
    // Allocate device memory
    float *deviceInput, *deviceKernel, *deviceOutput;
    cudaMalloc(&deviceInput, inputWidth * inputHeight * sizeof(float));
    cudaMalloc(&deviceKernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc(&deviceOutput, outputWidth * outputHeight * sizeof(float));
    
    // Copy data from host to device
    cudaMemcpy(deviceInput, hostInput.data(), inputWidth * inputHeight * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceKernel, hostKernel.data(), kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((outputWidth + TILE_SIZE - 1) / TILE_SIZE, 
                 (outputHeight + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel with timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    efficientConvolution2D<<<gridDim, blockDim>>>(
        deviceInput, 
        deviceKernel, 
        deviceOutput, 
        inputWidth, 
        inputHeight,
        outputWidth, 
        outputHeight
    );
    cudaEventRecord(stop);
    
    // Wait for kernel to finish
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(hostOutput.data(), deviceOutput, outputWidth * outputHeight * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print timing information
    std::cout << "Convolution completed in " << milliseconds << " ms" << std::endl;
    std::cout << "Input size: " << inputWidth << "x" << inputHeight << std::endl;
    std::cout << "Kernel size: " << kernelSize << "x" << kernelSize << std::endl;
    std::cout << "Output size: " << outputWidth << "x" << outputHeight << std::endl;
    
    // Print a small portion of the output for verification
    std::cout << "\nSample output values:" << std::endl;
    for (int i = 0; i < 5 && i < outputHeight; i++) {
        for (int j = 0; j < 5 && j < outputWidth; j++) {
            std::cout << hostOutput[i * outputWidth + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // Free device memory
    cudaFree(deviceInput);
    cudaFree(deviceKernel);
    cudaFree(deviceOutput);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}