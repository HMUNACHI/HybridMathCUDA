#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cufft.h>       // CUDA FFT library
#include <cublas_v2.h>   // CUDA BLAS library
#include <curand.h>      // CUDA random number generation library
#include <cusparse.h>    // CUDA sparse matrix library
#include <cusolverDn.h>  // CUDA solver library
#include <thrust/device_vector.h>  // Thrust template library
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

/*
CUDA Libraries Overview:
-----------------------
CUDA comes with several specialized libraries that provide optimized implementations of 
common mathematical and scientific computing operations. Using these libraries offers 
several advantages over writing custom CUDA kernels:

1. cuBLAS: Basic Linear Algebra Subroutines for high-performance matrix and vector operations
2. cuFFT: Fast Fourier Transform library for spectral analysis and signal processing
3. cuRAND: Random number generation library for statistical simulations and stochastic methods
4. cuSPARSE: Sparse matrix operations for efficiently handling large, sparse data structures
5. cuSOLVER: Dense and sparse direct solvers for linear systems, least squares, and eigenproblems
6. Thrust: High-level C++ template library providing parallel algorithms like sort, reduce, scan

These libraries are highly optimized for NVIDIA GPUs, using specialized hardware features
and memory access patterns to achieve peak performance. They abstract away the complex
implementation details while maintaining high performance.
*/

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(val) checkCudaError((val), #val, __FILE__, __LINE__)
inline void checkCudaError(cudaError_t result, const char* func, const char* file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << static_cast<unsigned int>(result) 
                  << " \"" << func << "\" \n" << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Print formatted matrix for display
void printMatrix(const float* matrix, int rows, int cols, const char* name) {
    std::cout << "Matrix " << name << " (" << rows << "x" << cols << "):" << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                      << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Demo 1: cuBLAS for matrix multiplication
void cublasDemoMatrixMultiply() {
    std::cout << "\n===== cuBLAS Demo: Matrix Multiplication =====" << std::endl;
    
    // Initialize matrices
    const int m = 4, n = 4, k = 4;  // Matrix dimensions
    float *h_A = new float[m * k];  // m x k matrix
    float *h_B = new float[k * n];  // k x n matrix
    float *h_C = new float[m * n];  // m x n matrix
    
    // Initialize input matrices
    for (int i = 0; i < m * k; i++) h_A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < k * n; i++) h_B[i] = static_cast<float>(i + 1);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, m * k * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, k * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, m * n * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Perform matrix multiplication: C = A * B
    // Note: cuBLAS uses column-major order, but we're using row-major,
    // so we compute B * A instead of A * B to get the correct result
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, m, k, 
                &alpha,
                d_B, n,    // B is treated as the first matrix
                d_A, k,    // A is treated as the second matrix
                &beta,
                d_C, n);   // Result in C
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results
    printMatrix(h_A, m, k, "A");
    printMatrix(h_B, k, n, "B");
    printMatrix(h_C, m, n, "C = A * B");
    
    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

// Demo 2: cuFFT for Fast Fourier Transform
void cufftDemoFFT() {
    std::cout << "\n===== cuFFT Demo: 1D Fast Fourier Transform =====" << std::endl;
    
    // Define signal parameters
    const int signal_size = 16;
    cufftComplex *h_signal = new cufftComplex[signal_size];
    cufftComplex *h_result = new cufftComplex[signal_size];
    
    // Create a simple sinusoidal signal
    for (int i = 0; i < signal_size; i++) {
        float x = static_cast<float>(i) / signal_size;
        h_signal[i].x = sin(2 * M_PI * 2 * x) + 0.5f * sin(2 * M_PI * 4 * x);  // Real part
        h_signal[i].y = 0.0f;  // Imaginary part (zero for real signal)
    }
    
    // Print original signal
    std::cout << "Original signal:" << std::endl;
    for (int i = 0; i < signal_size; i++) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                  << h_signal[i].x << " + " << h_signal[i].y << "i" << std::endl;
    }
    
    // Allocate device memory
    cufftComplex *d_signal, *d_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_signal, signal_size * sizeof(cufftComplex)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, signal_size * sizeof(cufftComplex)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_signal, h_signal, signal_size * sizeof(cufftComplex), cudaMemcpyHostToDevice));
    
    // Create cuFFT plan
    cufftHandle plan;
    cufftPlan1d(&plan, signal_size, CUFFT_C2C, 1);
    
    // Execute FFT
    cufftExecC2C(plan, d_signal, d_result, CUFFT_FORWARD);
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_result, signal_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    
    // Print FFT result
    std::cout << "\nFFT result:" << std::endl;
    for (int i = 0; i < signal_size; i++) {
        float magnitude = sqrt(h_result[i].x * h_result[i].x + h_result[i].y * h_result[i].y);
        std::cout << "Frequency " << i << ": Magnitude = " << magnitude 
                  << " (Real: " << h_result[i].x << ", Imag: " << h_result[i].y << ")" << std::endl;
    }
    
    // Execute inverse FFT to verify
    cufftExecC2C(plan, d_result, d_signal, CUFFT_INVERSE);
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_signal, d_signal, signal_size * sizeof(cufftComplex), cudaMemcpyDeviceToHost));
    
    // Normalize (cuFFT doesn't normalize)
    for (int i = 0; i < signal_size; i++) {
        h_signal[i].x /= signal_size;
        h_signal[i].y /= signal_size;
    }
    
    // Print inverse FFT result
    std::cout << "\nInverse FFT result (should match original signal):" << std::endl;
    for (int i = 0; i < signal_size; i++) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                  << h_signal[i].x << " + " << h_signal[i].y << "i" << std::endl;
    }
    
    // Clean up
    cufftDestroy(plan);
    cudaFree(d_signal);
    cudaFree(d_result);
    delete[] h_signal;
    delete[] h_result;
}

// Demo 3: cuRAND for random number generation
void curandDemoRandomNumbers() {
    std::cout << "\n===== cuRAND Demo: Random Number Generation =====" << std::endl;
    
    const int num_samples = 10;
    float *h_random = new float[num_samples];
    
    // Allocate device memory
    float *d_random;
    CHECK_CUDA_ERROR(cudaMalloc(&d_random, num_samples * sizeof(float)));
    
    // Create cuRAND generator
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    
    // Set seed
    curandSetPseudoRandomGeneratorSeed(generator, 1234ULL);
    
    // Generate random numbers (uniform distribution between 0 and 1)
    curandGenerateUniform(generator, d_random, num_samples);
    
    // Copy data back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_random, d_random, num_samples * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << "Uniform random numbers [0, 1]:" << std::endl;
    for (int i = 0; i < num_samples; i++) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(6) << h_random[i] << std::endl;
    }
    
    // Generate normally distributed random numbers
    curandGenerateNormal(generator, d_random, num_samples, 0.0f, 1.0f);
    
    // Copy data back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_random, d_random, num_samples * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print results
    std::cout << "\nNormally distributed random numbers (mean=0, stddev=1):" << std::endl;
    for (int i = 0; i < num_samples; i++) {
        std::cout << std::setw(10) << std::fixed << std::setprecision(6) << h_random[i] << std::endl;
    }
    
    // Clean up
    curandDestroyGenerator(generator);
    cudaFree(d_random);
    delete[] h_random;
}

// Demo 4: cuSPARSE for sparse matrix operations
void cusparseDemoSparseMatrix() {
    std::cout << "\n===== cuSPARSE Demo: Sparse Matrix-Vector Multiplication =====" << std::endl;
    
    // Create a sparse matrix in COO format (coordinate format)
    // Example 5x5 sparse matrix:
    /*
        10  0  0  0  0
         0 20  0  0  0
         0  0 30  0  0
         0  0  0 40  0
         0  0  0  0 50
    */
    
    const int num_rows = 5;
    const int num_cols = 5;
    const int num_non_zeros = 5;  // Only diagonal elements are non-zero
    
    // COO format components
    int *h_row_indices = new int[num_non_zeros]{0, 1, 2, 3, 4};  // Row indices
    int *h_col_indices = new int[num_non_zeros]{0, 1, 2, 3, 4};  // Column indices
    float *h_values = new float[num_non_zeros]{10.0f, 20.0f, 30.0f, 40.0f, 50.0f};  // Values
    
    // Dense vector x
    float *h_x = new float[num_cols]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    // Result vector y
    float *h_y = new float[num_rows]{0.0f};
    
    // Allocate device memory
    int *d_row_indices, *d_col_indices;
    float *d_values, *d_x, *d_y;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_row_indices, num_non_zeros * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_col_indices, num_non_zeros * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_values, num_non_zeros * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, num_cols * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, num_rows * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_row_indices, h_row_indices, num_non_zeros * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_col_indices, h_col_indices, num_non_zeros * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_values, h_values, num_non_zeros * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y, h_y, num_rows * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create cuSPARSE handle
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    
    // Create matrix descriptor
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    
    // Define alpha and beta scalars
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Create COO sparse matrix
    cusparseSpMatDescr_t matA;
    cusparseCooSetStridedBatch(matA, 1);
    cusparseCreateCoo(&matA, num_rows, num_cols, num_non_zeros,
                      d_row_indices, d_col_indices, d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    // Create dense vectors
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, num_cols, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, num_rows, d_y, CUDA_R_32F);
    
    // Allocate workspace
    size_t bufferSize = 0;
    void* dBuffer = NULL;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, vecX, &beta, vecY,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                            &bufferSize);
    
    cudaMalloc(&dBuffer, bufferSize);
    
    // Perform sparse matrix-vector multiplication: y = alpha * A * x + beta * y
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, vecX, &beta, vecY,
                 CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                 dBuffer);
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_y, d_y, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Print sparse matrix
    std::cout << "Sparse matrix A:" << std::endl;
    for (int i = 0; i < num_non_zeros; i++) {
        std::cout << "A[" << h_row_indices[i] << "][" << h_col_indices[i] << "] = " << h_values[i] << std::endl;
    }
    
    // Print vector x
    std::cout << "\nVector x:" << std::endl;
    for (int i = 0; i < num_cols; i++) {
        std::cout << "x[" << i << "] = " << h_x[i] << std::endl;
    }
    
    // Print result vector y = A * x
    std::cout << "\nResult vector y = A * x:" << std::endl;
    for (int i = 0; i < num_rows; i++) {
        std::cout << "y[" << i << "] = " << h_y[i] << std::endl;
    }
    
    // Clean up
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    
    cudaFree(dBuffer);
    cudaFree(d_row_indices);
    cudaFree(d_col_indices);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
    
    delete[] h_row_indices;
    delete[] h_col_indices;
    delete[] h_values;
    delete[] h_x;
    delete[] h_y;
}

// Demo 5: cuSOLVER for solving linear systems
void cusolverDemoLinearSolve() {
    std::cout << "\n===== cuSOLVER Demo: Solving Linear System Ax = b =====" << std::endl;
    
    // Define a linear system Ax = b
    /*
        3x + 1y - 2z = 5
        2x + 6y + 4z = 31
        1x + 1y + 8z = 17
    */
    
    const int n = 3;  // System size (3x3)
    
    // Matrix A (column-major for cuSOLVER)
    float h_A[n * n] = {
        3.0f, 2.0f, 1.0f,  // First column
        1.0f, 6.0f, 1.0f,  // Second column
        -2.0f, 4.0f, 8.0f  // Third column
    };
    
    // Right-hand side b
    float h_b[n] = {5.0f, 31.0f, 17.0f};
    
    // Solution vector x
    float h_x[n];
    
    // Allocate device memory
    float *d_A, *d_b, *d_work;
    int *d_info, h_info = 0;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, n * n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, n * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_info, sizeof(int)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create cuSOLVER handle
    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    
    // Calculate workspace size
    int workspace_size = 0;
    cusolverDnSgetrf_bufferSize(handle, n, n, d_A, n, &workspace_size);
    
    // Allocate workspace
    CHECK_CUDA_ERROR(cudaMalloc(&d_work, workspace_size * sizeof(float)));
    
    // LU factorization
    int *d_pivot;
    CHECK_CUDA_ERROR(cudaMalloc(&d_pivot, n * sizeof(int)));
    
    cusolverDnSgetrf(handle, n, n, d_A, n, d_work, d_pivot, d_info);
    
    // Check if factorization was successful
    CHECK_CUDA_ERROR(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (h_info != 0) {
        std::cerr << "LU factorization failed: " << h_info << std::endl;
    } else {
        // Solve the linear system
        cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, d_A, n, d_pivot, d_b, n, d_info);
        
        // Copy solution back to host
        CHECK_CUDA_ERROR(cudaMemcpy(h_x, d_b, n * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Print the matrix A
        std::cout << "Matrix A:" << std::endl;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                // Convert from column-major to row-major for printing
                std::cout << std::setw(10) << std::fixed << std::setprecision(2) 
                          << h_A[j * n + i] << " ";
            }
            std::cout << std::endl;
        }
        
        // Print the right-hand side b
        std::cout << "\nVector b:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(2) << h_b[i] << std::endl;
        }
        
        // Print the solution x
        std::cout << "\nSolution x:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "x[" << i << "] = " << std::setw(10) << std::fixed << std::setprecision(6) << h_x[i] << std::endl;
        }
        
        // Verify solution by calculating Ax
        std::cout << "\nVerifying solution (Ax should equal b):" << std::endl;
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                // Convert from column-major to row-major for calculation
                sum += h_A[j * n + i] * h_x[j];
            }
            std::cout << "Row " << i << ": " << std::setw(10) << std::fixed << std::setprecision(6) 
                      << sum << " (should be " << h_b[i] << ")" << std::endl;
        }
    }
    
    // Clean up
    cusolverDnDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_work);
    cudaFree(d_info);
    cudaFree(d_pivot);
}

// Demo 6: Thrust for parallel algorithms
void thrustDemoParallelAlgorithms() {
    std::cout << "\n===== Thrust Demo: Parallel Algorithms =====" << std::endl;
    
    const int size = 10;
    
    // Generate data on host
    std::vector<float> h_data(size);
    for (int i = 0; i < size; i++) {
        h_data[i] = static_cast<float>(rand() % 100);
    }
    
    // Print original data
    std::cout << "Original data:" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // Transfer data to device
    thrust::device_vector<float> d_data(h_data.begin(), h_data.end());
    
    // Sort data
    thrust::sort(d_data.begin(), d_data.end());
    
    // Transfer sorted data back to host
    thrust::copy(d_data.begin(), d_data.end(), h_data.begin());
    
    // Print sorted data
    std::cout << "\nSorted data:" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;
    
    // Find sum using reduction
    float sum = thrust::reduce(d_data.begin(), d_data.end(), 0.0f, thrust::plus<float>());
    
    // Find minimum and maximum values
    float min_val = thrust::reduce(d_data.begin(), d_data.end(), FLT_MAX, thrust::minimum<float>());
    float max_val = thrust::reduce(d_data.begin(), d_data.end(), -FLT_MAX, thrust::maximum<float>());
    
    // Print reduction results
    std::cout << "\nReduction results:" << std::endl;
    std::cout << "Sum: " << sum << std::endl;
    std::cout << "Min: " << min_val << std::endl;
    std::cout << "Max: " << max_val << std::endl;
    std::cout << "Average: " << sum / size << std::endl;
    
    // Create a new vector with transformed values (square each value)
    thrust::device_vector<float> d_squared(size);
    thrust::transform(d_data.begin(), d_data.end(), d_squared.begin(), thrust::square<float>());
    
    // Transfer squared data back to host
    std::vector<float> h_squared(size);
    thrust::copy(d_squared.begin(), d_squared.end(), h_squared.begin());
    
    // Print squared data
    std::cout << "\nSquared data:" << std::endl;
    for (int i = 0; i < size; i++) {
        std::cout << h_squared[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    // Show available CUDA devices
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "CUDA Libraries Demo" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "Found " << deviceCount << " CUDA-capable device(s)" << std::endl;
    
    // Print device properties
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        
        std::cout << "\nDevice " << i << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    }
    std::cout << std::endl;
    
    // Run library demos
    cublasDemoMatrixMultiply();
    cufftDemoFFT();
    curandDemoRandomNumbers();
    cusparseDemoSparseMatrix();
    cusolverDemoLinearSolve();
    thrustDemoParallelAlgorithms();
    
    std::cout << "\nAll demos completed successfully!" << std::endl;
    
    return 0;
}