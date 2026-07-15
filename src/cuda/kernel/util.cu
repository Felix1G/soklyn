#pragma once

#ifndef ACTIVATION_CU
#define ACTIVATION_CU

#include "types.hpp"
#include "math.cu"

__device__ inline void cast_f32_f16_t(const f32_t *in, f16_t *out, const uint32_t len) {
    if (const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len) {
        out[idx] = static_cast<f16_t>(in[idx]);
    }
}

__device__ inline void cast_f16_f32_t(const f16_t *in, f32_t *out, const uint32_t len) {
    if (const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len) {
        out[idx] = static_cast<f32_t>(in[idx]);
    }
}

__device__ inline uint32_t dev_nchw_idx(
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
 * 0: None, 1: Sigmoid, 2: ReLU, 3: LeakyReLU, 4: Tanh, 5: Softmax (Called using softmax_kernel), 6: SiLU, 7: Mish
 * @param act The activation mode.
 * @param sum The value to be passed into the activation function.
 * @param leaky_relu_coeff A coefficient specially for the LeakyReLU activation function as a multiplier for negative values.
 * @return The result of the activation function.
 */
__device__ inline f32_t dev_activation(const uint32_t act, const f32_t sum, const f32_t leaky_relu_coeff) {
    f32_t new_sum = sum;

    switch (act) {
        case 1:
            new_sum = 1.0f / (1.0f + CudaMath<f32_t>::exp(-sum));
            break;
        case 2:
            new_sum = CudaMath<f32_t>::max(0.0f, sum);
            break;
        case 3:
            new_sum = sum > 0.0f ? sum : leaky_relu_coeff * sum;
            break;
        case 4:
            new_sum = CudaMath<f32_t>::tanh(sum);
            break;
        case 6:
            new_sum = sum / (1.0f + CudaMath<f32_t>::exp(-sum));
            break;
        case 7: {
            // Upscaling to prevent overflow
            const f32_t e_x = CudaMath<f32_t>::exp(sum);

            const f32_t sp = sum > 20.0f ? sum : CudaMath<f32_t>::log(1.0f + e_x); // softplus

            const f32_t e_sp = CudaMath<f32_t>::exp(sp);
            const f32_t e_sp_inv = 1.0f / e_sp;
            const f32_t e_tanh = (e_sp - e_sp_inv) / (e_sp + e_sp_inv);

            const f32_t slope = sum * e_tanh;
            new_sum = slope;
            break;
        }
        default:
            new_sum = sum;
            break;
    }

    return new_sum;
}

/**
 * 0: None, 1: Sigmoid, 2: ReLU, 3: LeakyReLU, 4: Tanh, 5: Softmax (Directly paired with CEL in the compute error functions), 6: SiLU, 7: Mish
 * @tparam T Either f32_t or f16_t.
 * @param act The activation mode.
 * @param err_delta The error delta before activation.
 * @param out_v The result from the activation function.
 * @param in_x The input of the activation function.
 * @param leaky_relu_coeff A coefficient specially for the LeakyReLU activation function as a multiplier for negative values.
 * @return The new error delta with activation.
 */
__device__ inline f32_t dev_activation_derivative(
    const uint32_t act, const f32_t err_delta, const f32_t out_v, const f32_t in_x, const f32_t leaky_relu_coeff
) {
    f32_t err = err_delta;

    switch (act) {
        case 1:
            err *= out_v * (1.0f - out_v);
            break;
        case 2:
            err *= out_v > 0.0f ? 1.0f : 0.0f;
            break;
        case 3:
            err *= out_v > 0.0f ? 1.0f : leaky_relu_coeff;
            break;
        case 4:
            err *= 1.0f - out_v * out_v;
            break;
        case 6: {
            const f32_t sig_x = 1.0f / (1.0f + CudaMath<f32_t>::exp(-in_x));
            err *= sig_x + out_v * (1.0f - sig_x);
            break;
        }
        case 7: {
            const f32_t e_x = CudaMath<f32_t>::exp(in_x);

            const f32_t sp = in_x > 20.0f ? in_x : CudaMath<f32_t>::log(1.0f + e_x);
            const f32_t e_sp = CudaMath<f32_t>::exp(sp);
            const f32_t e_sp_inv = 1.0f / e_sp;

            const f32_t e_tanh = (e_sp - e_sp_inv) / (e_sp + e_sp_inv);
            const f32_t sig_x = e_x / (1.0f + e_x);
            const f32_t e_sech2 = 1.0f - e_tanh * e_tanh;

            const f32_t slope = e_tanh + in_x * sig_x * e_sech2;
            err *= slope;
            break;
        }
        default:
            err = err_delta;
            break;
    }

    return err;
}

/**
 * Applies the softmax activation function on the given matrix.
 * @tparam T Either f32_t or f16_t.
 * @param out The matrix where softmax will be applied per batch.
 * @param m The rows of the matrix (batch size).
 * @param n The columns of the matrix (features).
 */
// threadIdx.x corresponds to m (each thread handles ONE batch of n features)
template<typename T>
__device__ inline void softmax_kernel(T *out, const uint32_t m, const uint32_t n) {
    const uint32_t row = blockIdx.x;
    if (row >= m) return;

    __shared__ f32_t shared[32];
    const uint32_t lane = threadIdx.x % 32;
    const uint32_t wid = threadIdx.x / 32;
    const uint32_t num_warps = blockDim.x / 32;

    // ========================================================================
    // MAX VALUES (prevents overflow)
    // ========================================================================
    f32_t row_max = -INFINITY;
    for (uint32_t col = threadIdx.x; col < n; col += blockDim.x) {
        row_max = CudaMath<f32_t>::max(row_max, static_cast<f32_t>(out[row * n + col]));
    }

    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        row_max = CudaMath<f32_t>::max(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
    }

    if (lane == 0) shared[wid] = row_max; // warp maximum
    __syncthreads();

    row_max = threadIdx.x < num_warps ? shared[lane] : -INFINITY;
    if (wid == 0) {
        // block maximum
        for (uint32_t offset = 16; offset > 0; offset >>= 1) {
            row_max = CudaMath<f32_t>::max(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
        }
    }

    __shared__ f32_t block_global_max;
    if (threadIdx.x == 0) {
        block_global_max = row_max; // broadcasting
    }
    __syncthreads();

    // ========================================================================
    // SUM VALUES
    // ========================================================================
    f32_t local_sum = 0.0f;
    for (uint32_t col = threadIdx.x; col < n; col += blockDim.x) {
        local_sum += CudaMath<f32_t>::exp(static_cast<f32_t>(out[row * n + col]) - block_global_max);
    }

    for (uint32_t offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) shared[wid] = local_sum; // warp sum
    __syncthreads();

    local_sum = threadIdx.x < num_warps ? shared[lane] : 0.0f;
    if (wid == 0) {
        // block sum
        for (uint32_t offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }

    __shared__ f32_t block_global_sum;
    if (threadIdx.x == 0) {
        block_global_sum = local_sum; // broadcasting
    }
    __syncthreads();

    // ========================================================================
    // CALCULATE SOFTMAX
    // ========================================================================
    for (uint32_t col = threadIdx.x; col < n; col += blockDim.x) {
        const uint32_t idx = row * n + col;
        const T prev_sum = out[idx];
        const f32_t sum = CudaMath<f32_t>::exp(static_cast<f32_t>(prev_sum) - block_global_max) / block_global_sum;

        out[idx] = static_cast<T>(sum);
    }
}

template<typename T>
__device__ inline f32_t norm_derivative(
    const T * __restrict__ centered_out, const T * __restrict__ prenorm_out,
    const f32_t rstd, const f32_t err_delta, const f32_t n_w, f32_t &dNorm_w_acc, f32_t &dNorm_b_acc,
    const uint32_t norm, const uint32_t b_idx, const f32_t inv_n, const f32_t inv_m
) {
    f32_t d_prenorm = err_delta;

    if (norm == 1) {
        // RMSNorm
        const f32_t pval = static_cast<f32_t>(prenorm_out[b_idx]);

        dNorm_w_acc = err_delta * pval * rstd;

        d_prenorm = n_w * rstd * (err_delta - rstd * rstd * pval * pval * inv_n);
    } else if (norm == 2) {
        // LayerNorm
        const f32_t cval = static_cast<f32_t>(centered_out[b_idx]);

        dNorm_b_acc = err_delta;
        dNorm_w_acc = err_delta * cval * rstd;

        d_prenorm = n_w * rstd * (err_delta - err_delta * inv_n - rstd * rstd * cval * cval * err_delta * inv_n);
    } else if (norm == 3) {
        // BatchNorm
        const f32_t cval = static_cast<f32_t>(centered_out[b_idx]);

        dNorm_b_acc = err_delta;
        dNorm_w_acc = err_delta * cval * rstd;

        d_prenorm = n_w * rstd * (err_delta - err_delta * inv_m - rstd * rstd * cval * cval * err_delta * inv_m);
    }

    return d_prenorm;
}

template<typename T>
__device__ inline void dev_write_norm_gradients(
    f32_t *d_prenorm_out, f32_t *dNorm_w, f32_t *dNorm_b,
    const T * __restrict__ centered_out, const T * __restrict__ prenorm_out,
    const f32_t rstd, const f32_t err_delta, const f32_t n_w, const f32_t loss_scale,
    const uint32_t norm, const uint32_t idx,
    const f32_t inv_n, const f32_t inv_m
) {
    f32_t dNorm_w_acc = 0.0f;
    f32_t dNorm_b_acc = 0.0f;

    const f32_t d_prenorm = norm_derivative<T>(
        centered_out, prenorm_out, rstd, err_delta, n_w,
        dNorm_w_acc, dNorm_b_acc, norm, idx, inv_n, inv_m
    );

    d_prenorm_out[idx] = d_prenorm * loss_scale;
    dNorm_w[idx] = dNorm_w_acc;
    dNorm_b[idx] = dNorm_b_acc;
}

#endif
