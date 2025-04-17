#include <cuda_runtime.h>
#include <stdio.h>
#include "hermes_shm/data_structures/all.h"



// CUDA kernel for vector addition
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    // Calculate global thread ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Boundary check
    while (tid < N) {
        C[tid] = A[tid] + B[tid];
        // Handle multiple elements per thread using grid-stride loop
        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char **argv) {
    // Vector size
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < N; i++) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            break;
        }
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Vector addition completed successfully\n");
    return 0;
}