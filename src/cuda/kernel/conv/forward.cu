#pragma once

#ifndef CONV_FORWARD_CU
#define CONV_FORWARD_CU

#include "../util.cu"
#include "../math.cu"
#include "../ffn/forward.cu"

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
// threadIdx.x corresponds to output width, threadIdx.y corresponds to output height, threadIdx.z corresponds to batch size * output column
template<typename T>
__device__ inline void conv_forward_pass_0_kernel(
    T * __restrict__ prenorm_features, const T * __restrict__ in, const T * __restrict__ w, const T * __restrict__ b,
    const uint32_t use_bias, const uint32_t pad_mode, const uint32_t ic, const uint32_t oc,
    const uint32_t iw, const uint32_t ih, const uint32_t ow, const uint32_t oh, const uint32_t fw, const uint32_t fh,
    const uint32_t pad, const uint32_t stride_x, const uint32_t stride_y, const uint32_t dil_x, const uint32_t dil_y
) {
    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t bdim_x = blockDim.x;
    const uint32_t bdim_y = blockDim.y;
    const uint32_t blk_x = blockIdx.x;
    const uint32_t blk_y = blockIdx.y;

    const uint32_t tid = ty * bdim_x + tx;

    const uint32_t oh_idx = blk_y * bdim_y + ty;
    const uint32_t ow_idx = blk_x * bdim_x + tx;
    const uint32_t nc_idx = blockIdx.z;
    const uint32_t oc_idx = nc_idx % oc;
    const uint32_t batch_idx = nc_idx / oc;

    extern __shared__ char shared_mem[];
    T *shared_input_tile = reinterpret_cast<T *>(shared_mem);
    const uint32_t tile_h = (bdim_y - 1) * stride_y + (fh - 1) * dil_y + 1;
    const uint32_t tile_w = (bdim_x - 1) * stride_x + (fw - 1) * dil_x + 1;

    // Prevent bank conflict
    const uint32_t tile_stride = tile_w % 32 == 0 ? tile_w + 1 : tile_w;
    const uint32_t total_tile_elems = tile_stride * tile_h;

    const int32_t in_h_start = static_cast<int32_t>(blk_y * bdim_y * stride_y) - static_cast<int32_t>(pad);
    const int32_t in_w_start = static_cast<int32_t>(blk_x * bdim_x * stride_x) - static_cast<int32_t>(pad);

    T sum = static_cast<T>(0.0);

    for (uint32_t ic_idx = 0; ic_idx < ic; ++ic_idx) {
        const uint32_t threads_per_block = bdim_x * bdim_y;

        const T *in_ptr = in + (batch_idx * ic + ic_idx) * (ih * iw);

        // Read all the relevant input
        for (uint32_t tile_idx = tid; tile_idx < total_tile_elems; tile_idx += threads_per_block) {
            const uint32_t local_h = tile_idx / tile_stride;
            const uint32_t local_w = tile_idx % tile_stride;
            if (local_w >= tile_w) continue;

            int32_t gh = in_h_start + static_cast<int32_t>(local_h);
            int32_t gw = in_w_start + static_cast<int32_t>(local_w);

            if (gh >= 0 && gh < static_cast<int32_t>(ih) && gw >= 0 && gw < static_cast<int32_t>(iw)) {
                shared_input_tile[tile_idx] = in_ptr[static_cast<uint32_t>(gh) * iw + static_cast<uint32_t>(gw)];
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

                shared_input_tile[tile_idx] = fill_value;
            }
        }

        // Ensure inputs are all read
        __syncthreads();

        // Accumulate sum
        if (ow_idx < ow && oh_idx < oh) {
            const T *w_ptr = w + (oc_idx * ic + ic_idx) * (fh * fw);
            const uint32_t local_h_start = ty * stride_y;
            const uint32_t local_w_start = tx * stride_x;

#pragma unroll 4
            for (uint32_t fh_idx = 0; fh_idx < fh; ++fh_idx) {
                const uint32_t tile_row_offset = (local_h_start + fh_idx * dil_y) * tile_stride + local_w_start;
                const uint32_t weight_row_offset = fh_idx * fw;

                for (uint32_t fw_idx = 0; fw_idx < fw; ++fw_idx) {
                    sum += w_ptr[weight_row_offset + fw_idx] * shared_input_tile[tile_row_offset + fw_idx * dil_x];
                }
            }
        }

        // Finish this loop before the next buffer
        __syncthreads();
    }

    if (ow_idx < ow && oh_idx < oh) {
        if (use_bias) {
            sum += b[oc_idx];
        }

        const uint32_t out_idx = dev_nchw_idx(oc, oh, ow, batch_idx, oc_idx, oh_idx, ow_idx);
        prenorm_features[out_idx] = sum;
    }
}

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
__device__ inline void conv_forward_pass_1_kernel(
    T * __restrict__ preact_features, T * __restrict__ centered_features, const T * __restrict__ prenorm_features,
    const T * __restrict__ norm_w, const T * __restrict__ norm_b, T * __restrict__ norm_rstd,
    const uint32_t ow, const uint32_t oh, const uint32_t oc, const uint32_t on, const uint32_t norm
) {
    if (norm == 1 || norm == 2) {
        // RMSNorm vs LayerNorm
        forward_pass_1_kernel<T>(
            preact_features, centered_features, prenorm_features, norm_w, norm_b, norm_rstd,
            on, ow * oh * oc, norm
        );
    } else {
        // BatchNorm
        forward_pass_1_kernel<T>(
            preact_features, centered_features, prenorm_features, norm_w, norm_b, norm_rstd,
            on * oh * ow, oc, norm, true, oh, ow
        );
    }
}

template<typename T>
__device__ inline void conv_forward_pass_2_kernel(
    T * __restrict__ features, T * __restrict__ predrop_features, T * __restrict__ prepooling_features,
    const T * __restrict__ preact_features, T * __restrict__ mask,
    const uint32_t use_pooling, const uint32_t pad_mode, const uint32_t ic, const uint32_t oc,
    const uint32_t iw, const uint32_t ih, const uint32_t ow, const uint32_t oh, const uint32_t pw, const uint32_t ph,
    const uint32_t pad, const uint32_t stride_x, const uint32_t stride_y, const uint32_t dil_x, const uint32_t dil_y
) {

}

#endif
