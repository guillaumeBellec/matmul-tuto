// matmul.c — Naive and cache-friendly matrix multiplication
// Compiled as a shared library, called from Python via ctypes
//
// Compile: gcc -O2 -shared -fPIC -o matmul.so matmul.c

#include <stddef.h>

// Naive ijk order: A[i][k] accessed sequentially, but B[k][j] jumps across rows
// → cache-unfriendly access on B
void matmul_naive(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Cache-friendly ikj order: inner loop walks B[k][j] and C[i][j] sequentially
// → both B and C are accessed in row-major order → much better cache utilization
void matmul_cache_friendly(const float *A, const float *B, float *C, int M, int N, int K) {
    // Zero out C
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

// Tiled version: processes TILE×TILE blocks to maximize L1 cache reuse
#define TILE 32

void matmul_tiled(const float *A, const float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    for (int i0 = 0; i0 < M; i0 += TILE) {
        for (int k0 = 0; k0 < K; k0 += TILE) {
            for (int j0 = 0; j0 < N; j0 += TILE) {
                int i_end = i0 + TILE < M ? i0 + TILE : M;
                int k_end = k0 + TILE < K ? k0 + TILE : K;
                int j_end = j0 + TILE < N ? j0 + TILE : N;
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = A[i * K + k];
                        for (int j = j0; j < j_end; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
