#pragma once

#ifndef FFN_FORWARD_CU
#define FFN_FORWARD_CU

#include "forward.cuh"

template<typename T>
__device__ void forward_pass_0_kernel(
    T * __restrict__ prenorm_out, const T * __restrict__ in, const T * __restrict__ w, const T * __restrict__ b,
    const uint32_t use_bias, const uint32_t m, const uint32_t n, const uint32_t wc, const uint32_t tile_dim
) {
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Weight multiplication (due to thread syncing in dev_gemm, it must be called outside the if statement)
    T sum = dev_gemm<T>(in, w, m, n, wc, tile_dim, row, col);

    if (row < m && col < wc) {
        // Bias addition
        if (use_bias) {
            sum += b[col];
        }

        const uint32_t idx = row * wc + col;
        prenorm_out[idx] = sum;
    }
}

// Each block owns one entire row (norm != BatchNorm) or column (norm == BatchNorm)
template<typename T>
__device__ void forward_pass_1_kernel(
    T * __restrict__ preact_out, T * __restrict__ centered_out, const T * __restrict__ prenorm_out,
    const T * __restrict__ norm_w, const T * __restrict__ norm_b, T * __restrict__ norm_rstd,
    const uint32_t m, const uint32_t wc, const uint32_t norm,
    const bool use_batch_nchw, const uint32_t oh, const uint32_t ow
) {
    extern __shared__ f32_t shared_sum[];
    const uint32_t bid = blockIdx.x;
    if ((norm == 3 && bid >= wc) || (norm != 3 && bid >= m)) return;

    const uint32_t tid = threadIdx.x;

    if (norm == 1) {
        // RMSNorm
        const uint32_t row = bid;
        f32_t sq_sum = 0.0f;

        // include values where indices exceed this block size
        for (uint32_t col = tid; col < wc; col += blockDim.x) {
            const f32_t val = static_cast<f32_t>(prenorm_out[row * wc + col]);
            sq_sum += val * val; // Squaring following RMS formula
        }

        // Reduce sums to index 0
        shared_sum[tid] = sq_sum;
        dev_block_stride_sum_1d(shared_sum);

        // Finish calculation
        __syncthreads();
        const f32_t rms_variance = shared_sum[0] / static_cast<f32_t>(wc) + 1e-6f;
        const f32_t rstd_f32 = CudaMath<f32_t>::rsqrt(rms_variance);
        const T rstd = static_cast<T>(rstd_f32);

        if (tid == 0) {
            norm_rstd[bid] = rstd;
        }

        for (uint32_t col = tid; col < wc; col += blockDim.x) {
            const uint32_t idx = row * wc + col;
            centered_out[idx] = prenorm_out[idx];
            preact_out[idx] = prenorm_out[idx] * rstd * norm_w[col];
        }
    } else if (norm == 2 || norm == 3) {
        // LayerNorm or BatchNorm
        const uint32_t n = norm == 2 ? wc : m;
        const uint32_t stride = blockDim.x;

        // ------------- FIND MEAN -------------
        f32_t total_sum = 0.0f;

        // include values where indices exceed this block size
        for (uint32_t i = tid; i < n; i += stride) {
            const uint32_t idx = get_norm_data_idx(use_batch_nchw, oh, ow, bid, wc, i, norm);
            total_sum += static_cast<f32_t>(prenorm_out[idx]);
        }

        shared_sum[tid] = total_sum;
        dev_block_stride_sum_1d(shared_sum);

        __syncthreads();
        const f32_t mean = shared_sum[0] / static_cast<f32_t>(n);
        __syncthreads(); // prevent mean from being overwritten

        // ------------- FIND VARIANCE -------------
        f32_t std_sum = 0.0f;

        // include values where indices exceed this block size
        for (uint32_t i = tid; i < n; i += stride) {
            const uint32_t idx = get_norm_data_idx(use_batch_nchw, oh, ow, bid, wc, i, norm);
            const f32_t val = static_cast<f32_t>(prenorm_out[idx]) - mean;
            std_sum += val * val;
        }

        shared_sum[tid] = std_sum;
        dev_block_stride_sum_1d(shared_sum);

        // ------------- FINISH CALCULATIONS -------------
        __syncthreads();
        const f32_t variance = shared_sum[0] / static_cast<f32_t>(n) + 1e-6f;
        const f32_t rstd_f32 = CudaMath<f32_t>::rsqrt(variance);
        const T rstd = static_cast<T>(rstd_f32);

        if (tid == 0) {
            norm_rstd[bid] = rstd;
        }

        for (uint32_t i = tid; i < n; i += stride) {
            const uint32_t idx = get_norm_data_idx(use_batch_nchw, oh, ow, bid, wc, i, norm);
            const uint32_t col = norm == 2 ? i : bid;
            const f32_t val_f32 = static_cast<f32_t>(prenorm_out[idx]) - mean;
            centered_out[idx] = static_cast<T>(val_f32);
            preact_out[idx] = static_cast<T>(val_f32 * rstd_f32 * static_cast<f32_t>(norm_w[col]) +
                                static_cast<f32_t>(norm_b[col]));
        }
    }
}

// threadIdx.x corresponds to wc (features), threadIdx.y corresponds to m (batch size)
template<typename T>
__device__ void forward_pass_2_kernel(
    T * __restrict__ out, T * __restrict__ predrop_out, const T * __restrict__ preact_out, T * __restrict__ mask,
    const uint32_t m, const uint32_t n, const uint32_t act,
    const f32_t leaky_relu_coeff, const uint32_t use_dropout, const f32_t mask_coeff, const uint32_t seed
) {
    const T zero = static_cast<T>(0.0f);
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; row < m && col < n) {
        const uint32_t idx = row * n + col;
        f32_t sum = static_cast<f32_t>(preact_out[idx]);

        // activation function
        sum = dev_activation(act, sum, leaky_relu_coeff);
        predrop_out[idx] = static_cast<T>(sum);

        // dropout
        if (use_dropout) {
            if (dev_gen_random_f32_t(col, row, seed + row * 7) >= mask_coeff) {
                const f32_t scale = 1.0f / (1.0f - mask_coeff);
                mask[idx] = static_cast<T>(scale);
                out[idx] = static_cast<T>(sum * scale);
            } else {
                mask[idx] = zero;
                out[idx] = zero;
            }
        } else {
            out[idx] = static_cast<T>(sum);
        }
    }
}

#endif
