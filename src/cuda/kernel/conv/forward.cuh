#pragma once

#ifndef CONV_FORWARD_CUH
#define CONV_FORWARD_CUH

#include "../util.cuh"
#include "../math.cuh"
#include "../ffn/forward.cuh"

template<typename T>
__device__ __forceinline__ void dev_conv_read_to_shared_memory(
    T* __restrict__ shared_input_tile, const T* __restrict__ in_ptr,
    uint32_t tid, uint32_t total_tile_elems, uint32_t threads_per_block,
    uint32_t tile_stride, uint32_t tile_w, uint32_t in_h_start, uint32_t in_w_start,
    uint32_t ih, uint32_t iw, uint32_t pad, uint32_t pad_mode,
    uint32_t act = 0, f32_t leaky_relu_coeff = 0.0f, T* __restrict__ postact_features = nullptr
) {
    for (uint32_t tile_idx = tid; tile_idx < total_tile_elems; tile_idx += threads_per_block) {
        uint32_t local_h = tile_idx / tile_stride;
        uint32_t local_w = tile_idx % tile_stride;
        if (local_w >= tile_w) continue;

        int32_t gh = in_h_start + static_cast<int32_t>(local_h);
        int32_t gw = in_w_start + static_cast<int32_t>(local_w);

        T read_val;

        if (gh >= 0 && gh < static_cast<int32_t>(ih) && gw >= 0 && gw < static_cast<int32_t>(iw)) {
            read_val = in_ptr[static_cast<uint32_t>(gh) * iw + static_cast<uint32_t>(gw)];
        } else {
            T fill_value = static_cast<T>(0);

            const int32_t max_h_pad = static_cast<int32_t>(ih) + static_cast<int32_t>(pad);
            const int32_t max_w_pad = static_cast<int32_t>(iw) + static_cast<int32_t>(pad);

            if (gh >= -static_cast<int32_t>(pad) && gh < max_h_pad &&
                gw >= -static_cast<int32_t>(pad) && gw < max_w_pad) {
                if (pad_mode == 1) {
                    // Reflective
                    gh = gh < 0
                             ? -gh
                             : gh >= static_cast<int32_t>(ih)
                                   ? 2 * (static_cast<int32_t>(ih) - 1) - gh
                                   : gh;
                    gw = gw < 0
                             ? -gw
                             : gw >= static_cast<int32_t>(iw)
                                   ? 2 * (static_cast<int32_t>(iw) - 1) - gw
                                   : gw;
                    fill_value = in_ptr[static_cast<uint32_t>(gh) * iw + static_cast<uint32_t>(gw)];
                } else if (pad_mode == 2) {
                    // Replicate
                    gh = gh < 0 ? 0 : gh >= static_cast<int32_t>(ih) ? static_cast<int32_t>(ih) - 1 : gh;
                    gw = gw < 0 ? 0 : gw >= static_cast<int32_t>(iw) ? static_cast<int32_t>(iw) - 1 : gw;
                    fill_value = in_ptr[static_cast<uint32_t>(gh) * iw + static_cast<uint32_t>(gw)];
                }
            }

            read_val = fill_value;
        }

        // activation is not softmax
        if (act != 5 && gh >= 0 && gh < static_cast<int32_t>(ih) && gw >= 0 && gw < static_cast<int32_t>(iw)) {
            read_val = static_cast<T>(dev_activation(act, static_cast<f32_t>(read_val), leaky_relu_coeff));
            postact_features[static_cast<uint32_t>(gh) * iw + static_cast<uint32_t>(gw)] = read_val;
        }

        shared_input_tile[tile_idx] = read_val;
    }
}

/**
 * The first part of the forward pass. This function handles the CNN linear calculations (wx + b).
 *
 * Each thread loads the relevant part of the input in shared memory with a stride of
 * the number of threads per block such that no single element is fetched/read twice from VRAM.
 *
 * Suggested tile dimension to be 16 x 16.
 *
 * input ---linear---> prenorm_features
 *
 * @tparam T Either f32_t or f16_t.
 * @param prenorm_features The tensor representing the features before normalisation (after linear).
 * @param in The input tensor.
 * @param w The filter weight tensor.
 * @param b The filter bias tensor.
 * @param use_bias If true, bias is added. Otherwise, bias is not added.
 * @param pad_mode The mode of padding used.
 * @param ic The number of input channels (channels of the incoming tensor).
 * @param oc The number of output channels (number of feature maps to produce).
 * @param iw The width of the input tensor.
 * @param ih The height of the input tensor.
 * @param ow The width of the output feature map.
 * @param oh The height of the output feature map.
 * @param fw The width of the filter kernel.
 * @param fh The height of the filter kernel.
 * @param pad The padding size.
 * @param stride_x The stride step size along the width (horizontal axis).
 * @param stride_y The stride step size along the height (vertical axis).
 * @param dil_x The dilation in the horizontal axis.
 * @param dil_y The dilation in the vertical axis.
 */
template<typename T>
__device__ void conv_forward_pass_0_kernel(
    T * __restrict__ prenorm_features, const T * __restrict__ in, const T * __restrict__ w, const T * __restrict__ b,
    uint32_t use_bias, uint32_t ic, uint32_t oc,
    uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh, uint32_t fw, uint32_t fh,
    uint32_t pad, uint32_t pad_mode,
    uint32_t stride_x, uint32_t stride_y, uint32_t dil_x, uint32_t dil_y
);

/**
 * The second part of the forward pass. This function handles the normalisation calculations.
 *
 * prenorm ---centering---> centered ---normalisation---> preact
 *
 * 1: RMSNorm, 2: LayerNorm, 3: BatchNorm
 *
 * @tparam T Either f32_t or f16_t.
 * @param preact_features The tensor representing the output before activation (after normalisation).
 * @param centered_features The tensor representing the centered values (x - mean).
 * @param prenorm_features The tensor representing the output before normalisation.
 * @param norm_w The tensor representing the weights of normalisation.
 * @param norm_b The tensor representing the biases of normalisation (ignored in RMSNorm).
 * @param norm_rstd The tensor representing the reciprocal of standard deviation.
 * @param ow The spatial width of the output tensors.
 * @param oh The spatial height of the output tensors.
 * @param oc The channels of the output tensors.
 * @param on The batch size of the output tensors.
 * @param norm The normalisation mode.
 */
template<typename T>
__device__ void conv_forward_pass_1_kernel(
    T * __restrict__ preact_features, T * __restrict__ centered_features, const T * __restrict__ prenorm_features,
    const T * __restrict__ norm_w, const T * __restrict__ norm_b, T * __restrict__ norm_rstd,
    uint32_t ow, uint32_t oh, uint32_t oc, uint32_t on, uint32_t norm
);

/**
 * The final part of the forward pass. This function handles the activation and dropout.
 *
 * preact ---activation---> prepooling ---pooling---> predrop ---dropout/masking---> out
 *
 * preact ---activation-and-pooling-DISABLED---> predrop ---dropout/masking---> out
 *
 * @tparam T Either f32_t or f16_t.
 * @param features The final output tensor.
 * @param predrop_features The tensor before dropout (after activation).
 * @param prepooling_features The tensor representing the output before activation.
 * @param preact_features The tensor representing the output before activation.
 * @param mask The masking tensor for dropout.
 * @param use_dropout If set to false, dropouts are disabled.
 * @param pool_mode The mode of pooling used.
 * @param pad The padding size.
 * @param pad_mode The mode of padding used.
 * @param channels The number of output channels (number of feature maps to produce).
 * @param iw The width of the input tensor.
 * @param ih The height of the input tensor.
 * @param ow The width of the output feature map.
 * @param oh The height of the output feature map.
 * @param pw The width of the pooling kernel.
 * @param ph The height of the pooling kernel.
 * @param act The activation mode.
 * @param leaky_relu_coeff A coefficient specially for the LeakyReLU activation function as a multiplier for negative values.
 * @param mask_coeff The mask value (0.0 - 1.0) where 0.0 means no neurons are dropped and 1.0 means all neurons are dropped.
 * @param seed The seed that will be passed to an RNG.
 * @param stride_x The stride step size along the width (horizontal axis).
 * @param stride_y The stride step size along the height (vertical axis).
 * @param dil_x The dilation in the horizontal axis.
 * @param dil_y The dilation in the vertical axis.
 */
template<typename T>
__device__ void conv_forward_pass_2_kernel(
    T* __restrict__ features, T* __restrict__ predrop_features, T* __restrict__ prepooling_features,
    const T* __restrict__ preact_features, T* __restrict__ mask, uint32_t use_dropout,
    uint32_t pool_mode, uint32_t channels,
    uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh, uint32_t pw, uint32_t ph,
    uint32_t pad, uint32_t pad_mode,
    uint32_t stride_x, uint32_t stride_y, uint32_t dil_x, uint32_t dil_y,
    uint32_t act, f32_t leaky_relu_coeff, f32_t mask_coeff, uint32_t seed
);

#endif