#pragma once

#ifndef FFN_FORWARD_CUH
#define FFN_FORWARD_CUH

#include "../util.cuh"
#include "../math.cuh"

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
__device__ void forward_pass_0_kernel(
    T * __restrict__ prenorm_out, const T * __restrict__ in, const T * __restrict__ w, const T * __restrict__ b,
    uint32_t use_bias, uint32_t m, uint32_t n, uint32_t wc, uint32_t tile_dim
);

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
template<typename T>
__device__ void forward_pass_1_kernel(
    T * __restrict__ preact_out, T * __restrict__ centered_out, const T * __restrict__ prenorm_out,
    const T * __restrict__ norm_w, const T * __restrict__ norm_b, T * __restrict__ norm_rstd,
    uint32_t m, uint32_t wc, uint32_t norm,
    bool use_batch_nchw = false, uint32_t oh = 0, uint32_t ow = 0
);

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
template<typename T>
__device__ void forward_pass_2_kernel(
    T * __restrict__ out, T * __restrict__ predrop_out, const T * __restrict__ preact_out, T * __restrict__ mask,
    uint32_t m, uint32_t n, uint32_t act,
    f32_t leaky_relu_coeff, uint32_t use_dropout, f32_t mask_coeff, uint32_t seed
);

#if __CUDA_ARCH__ >= 700
__device__ void forward_pass_0_wmma_kernel(
    f16_t* prenorm_out, const f16_t* in, const f16_t* w, const f16_t* b,
    uint32_t use_bias, uint32_t m, uint32_t n, uint32_t wc
);
#endif

#endif