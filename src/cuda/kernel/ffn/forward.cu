#pragma once

#ifndef FFN_FORWARD_CU
#define FFN_FORWARD_CU

#include "../util.cu"
#include "../math.cu"

/**
 * The first part of the forward pass. This function handles the linear calculations (wx + b).
 *
 * input ---linear---> prenorm
 *
 * @tparam T Either f32_t or f16_t.
 * @param prenorm_out The matrix representing the output before normalisation (after linear).
 * @param in The input matrix.
 * @param w The weight matrix.
 * @param b The bias matrix.
 * @param use_bias If true, bias is added. Otherwise, bias is not added.
 * @param m The rows of the input matrix (batch size).
 * @param n The columns of the input matrix (input features).
 * @param wc The columns of the weight matrix (output features).
 * @param tile_dim The dimension (both width and height) of the 2D thread block and data tile.
 */
// threadIdx.x corresponds to wc (features), threadIdx.y corresponds to m (batch size)
template<typename T>
__device__ inline void forward_pass_0_kernel(
    T * __restrict__ prenorm_out, const T * __restrict__ in, const T * __restrict__ w, const T * __restrict__ b,
    const uint32_t use_bias, const uint32_t m, const uint32_t n, const uint32_t wc,
    const uint32_t tile_dim
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

__device__ __forceinline__ uint32_t get_norm_data_idx(
    const bool use_batch_nchw, const uint32_t oh, const uint32_t ow,
    const uint32_t bid, const uint32_t wc, const uint32_t i, const uint32_t norm) {
    if (use_batch_nchw) {
        const uint32_t batch_idx = i / (oh * ow);
        const uint32_t spatial_idx = i - batch_idx * (oh * ow);
        return dev_nchw_idx(wc, oh, ow, batch_idx, bid, spatial_idx / ow, spatial_idx % ow);
    }

    return norm == 2 ? bid * wc + i : i * wc + bid;
}

/**
 * The second part of the forward pass. This function handles the normalisation calculations.
 *
 * prenorm ---centering---> centered ---normalisation---> preact
 *
 * 1: RMSNorm, 2: LayerNorm, 3: BatchNorm
 *
 * @tparam T Either f32_t or f16_t.
 * @param preact_out The matrix representing the output before activation (after normalisation).
 * @param centered_out The matrix representing the centered values (x - mean).
 * @param prenorm_out The matrix representing the output before normalisation.
 * @param norm_w The matrix representing the weights of normalisation.
 * @param norm_b The matrix representing the biases of normalisation (ignored in RMSNorm).
 * @param norm_rstd The matrix representing the reciprocal of standard deviation.
 * @param m The rows of the output matrix (batch size).
 * @param wc The columns of the output matrix or the norm weights (output features).
 * @param norm The normalisation mode.
 * @param use_batch_nchw Specially for the CNN forward norm, which is set true for BatchNorm to use NCHW indexing.
 * Output channel is determined from wc. Batch is determined from m / (oh * ow).
 * @param oh The CNN output tensor's height.
 * @param ow The CNN output tensor's width.
 */
// Each block owns one entire row (norm != BatchNorm) or column (norm == BatchNorm)
template<typename T>
__device__ inline void forward_pass_1_kernel(
    T * __restrict__ preact_out, T * __restrict__ centered_out, const T * __restrict__ prenorm_out,
    const T * __restrict__ norm_w, const T * __restrict__ norm_b, T * __restrict__ norm_rstd,
    const uint32_t m, const uint32_t wc, const uint32_t norm,
    const bool use_batch_nchw = false, const uint32_t oh = 0, const uint32_t ow = 0
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

/**
 * The final part of the forward pass. This function handles the activation and dropout.
 *
 * preact ---activation---> predrop ---dropout/masking---> out
 * @tparam T Either f32_t or f16_t.
 * @param out The final output matrix.
 * @param predrop_out The matrix representing the matrix before dropout (after activation).
 * @param preact_out The matrix representing the output before activation.
 * @param mask The masking matrix for dropout.
 * @param m The rows of the output matrix (batch size).
 * @param n The columns of the output matrix (output features).
 * @param act The activation mode.
 * @param leaky_relu_coeff A coefficient specially for the LeakyReLU activation function as a multiplier for negative values.
 * @param use_dropout If set to false, dropouts are disabled.
 * @param mask_coeff The mask value (0.0 - 1.0) where 0.0 means no neurons are dropped and 1.0 means all neurons are dropped.
 * @param seed The seed that will be passed to an RNG.
 */
// threadIdx.x corresponds to wc (features), threadIdx.y corresponds to m (batch size)
template<typename T>
__device__ inline void forward_pass_2_kernel(
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
