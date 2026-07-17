#pragma once

#ifndef UTIL_CUH
#define UTIL_CUH

#include "types.cuh"
#include "math.cuh"

__device__ __forceinline__ void cast_f32_f16_t(const f32_t *in, f16_t *out, const uint32_t len) {
    if (const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len) {
        out[idx] = static_cast<f16_t>(in[idx]);
    }
}

__device__ __forceinline__ void cast_f16_f32_t(const f16_t *in, f32_t *out, const uint32_t len) {
    if (const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len) {
        out[idx] = static_cast<f32_t>(in[idx]);
    }
}

__device__ __forceinline__ uint32_t dev_nchw_idx(
    const uint32_t c, const uint32_t h, const uint32_t w,
    const uint32_t ni, const uint32_t ci, const uint32_t hi, const uint32_t wi
) {
    return (ni * c + ci) * (h * w) + hi * w + wi;
}

/**
 * Tree-based parallel reduction: sums the entire shared_sum array into shared_sum[0].
 *
 * This is 1 dimensional, summing across the x-axis.
 */
__device__ inline void dev_block_stride_sum_1d(f32_t *shared_sum) {
    const uint32_t tid = threadIdx.x;
    __syncthreads(); // ensure shared_sum has been fully written
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared_sum[tid] += shared_sum[tid + stride];
        __syncthreads();
    }
}

/**
 * @param act The activation mode.
 * @param sum The value to be passed into the activation function.
 * @param leaky_relu_coeff A coefficient specially for the LeakyReLU activation function as a multiplier for negative values.
 * @return The result of the activation function.
 */
__device__ f32_t dev_activation(uint32_t act, f32_t sum, f32_t leaky_relu_coeff);

/**
 * @tparam T Either f32_t or f16_t.
 * @param act The activation mode.
 * @param err_delta The error delta before activation.
 * @param out_v The result from the activation function.
 * @param in_x The input of the activation function.
 * @param leaky_relu_coeff A coefficient specially for the LeakyReLU activation function as a multiplier for negative values.
 * @return The new error delta with activation.
 */
__device__ f32_t dev_activation_derivative(
    uint32_t act, f32_t err_delta, f32_t out_v, f32_t in_x, f32_t leaky_relu_coeff
);

/**
 * Applies the softmax activation function on the given matrix.
 * @tparam T Either f32_t or f16_t.
 * @param out The matrix where softmax will be applied per batch.
 * @param m The rows of the matrix (batch size).
 * @param n The columns of the matrix (features).
 */
// threadIdx.x corresponds to m (each thread handles ONE batch of n features)
template<typename T>
__device__ void softmax_kernel(T *out, uint32_t m, uint32_t n);

template<typename T>
__device__ void dev_write_norm_gradients(
    f32_t *d_prenorm_out, f32_t *dNorm_w, f32_t *dNorm_b,
    const T * __restrict__ centered_out, const T * __restrict__ prenorm_out,
    f32_t rstd, f32_t err_delta, f32_t n_w, f32_t loss_scale,
    uint32_t norm, uint32_t idx, f32_t inv_n, f32_t inv_m
);

#endif