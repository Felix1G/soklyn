#pragma once

#ifndef MATH_CU
#define MATH_CU

#include "math.cuh"

template<typename T>
__device__ T dev_gemm(
    const T * __restrict__ a, const T * __restrict__ b,
    const uint32_t m, const uint32_t n, const uint32_t p,
    const uint32_t tile_dim, const uint32_t row, const uint32_t col
) {
    extern __shared__ uint8_t gemm_shared_mem[];

    T * __restrict__ tile_A = reinterpret_cast<T *>(gemm_shared_mem);
    T * __restrict__ tile_B = tile_A + tile_dim * tile_dim;

    f32_t sum_f32 = 0.0f;

    for (uint32_t t = 0; t < (n + tile_dim - 1) / tile_dim; ++t) {
        const uint32_t tile_offset = t * tile_dim;
        const uint32_t tid = threadIdx.y * tile_dim + threadIdx.x;

        if (const uint32_t tpos_x = tile_offset + threadIdx.x; row < m && tpos_x < n) {
            tile_A[tid] = a[row * n + tpos_x];
        } else {
            tile_A[tid] = static_cast<T>(0.0f);
        }

        if (const uint32_t tpos_y = tile_offset + threadIdx.y; col < p && tpos_y < n) {
            tile_B[tid] = b[tpos_y * p + col];
        } else {
            tile_B[tid] = static_cast<T>(0.0f);
        }

        __syncthreads();

        // Dot product
        for (uint32_t k = 0; k < tile_dim; ++k) {
            const f32_t val_a = static_cast<f32_t>(tile_A[threadIdx.y * tile_dim + k]);
            const f32_t val_b = static_cast<f32_t>(tile_B[k * tile_dim + threadIdx.x]);
            sum_f32 += val_a * val_b;
        }

        __syncthreads();
    }

    return static_cast<T>(sum_f32);
}

template<typename T>
__device__ void gemm_kernel(
    const T * __restrict__ a, const T * __restrict__ b, T * __restrict__ c,
    const uint32_t m, const uint32_t n, const uint32_t p, const uint32_t tile_dim
) {
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const T sum = dev_gemm(a, b, m, n, p, tile_dim, row, col);

    if (row < m && col < p) {
        c[row * p + col] = sum;
    }
}

template<typename T>
__device__ void geam_kernel(const T * __restrict__ a, const T * __restrict__ b, T * __restrict__ c,
                                   const uint32_t m, const uint32_t n) {
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; row < m && col < n) {
        const uint32_t idx = row * n + col;
        c[idx] = a[idx] + b[idx];
    }
}

#endif
