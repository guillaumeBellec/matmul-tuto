// matmul_cuda.cu — Minimal CUDA matrix multiplication kernels
// Compiled as a shared library, called from Python via ctypes
//
// Compile: nvcc -shared -Xcompiler -fPIC -o matmul_cuda.so matmul_cuda.cu

#include <cuda_runtime.h>
#include <stdio.h>

// Kernel 1: Naive — one thread computes one element of C
__global__ void matmul_naive_kernel(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Kernel 2: Shared-memory tiled — blocks cooperatively load tiles into shared memory
#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C,
                                     int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tile of A into shared memory
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host wrapper: allocates device memory, runs kernel, copies result back
extern "C" void cuda_matmul_naive(const float *h_A, const float *h_B, float *h_C,
                                   int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_naive_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

extern "C" void cuda_matmul_tiled(const float *h_A, const float *h_B, float *h_C,
                                   int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
