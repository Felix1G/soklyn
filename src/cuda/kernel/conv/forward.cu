#pragma once

#ifndef CONV_FORWARD_CU
#define CONV_FORWARD_CU

#include "forward.cuh"

// threadIdx.x corresponds to output width, threadIdx.y corresponds to output height, threadIdx.z corresponds to batch size * output column
template<typename T>
__device__ void conv_forward_pass_0_kernel(
    T * __restrict__ prenorm_features, const T * __restrict__ in, const T * __restrict__ w, const T * __restrict__ b,
    const uint32_t use_bias, const uint32_t ic, const uint32_t oc,
    const uint32_t iw, const uint32_t ih, const uint32_t ow, const uint32_t oh, const uint32_t fw, const uint32_t fh,
    const uint32_t pad, const uint32_t pad_mode,
    const uint32_t stride_x, const uint32_t stride_y, const uint32_t dil_x, const uint32_t dil_y
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
        dev_conv_read_to_shared_memory<T>(
            shared_input_tile, in_ptr, tid, total_tile_elems, threads_per_block, tile_stride,
            tile_w, in_h_start, in_w_start, ih, iw, pad, pad_mode
        );

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

template<typename T>
__device__ void conv_forward_pass_1_kernel(
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

// threadIdx.x corresponds to output width, threadIdx.y corresponds to output height, threadIdx.z corresponds to batch size * output column
template<typename T>
__device__ void conv_forward_pass_2_kernel(
    T* __restrict__ features, T* __restrict__ predrop_features, T* __restrict__ prepooling_features,
    const T* __restrict__ preact_features, T* __restrict__ mask, const uint32_t use_dropout,
    const uint32_t pool_mode, const uint32_t channels,
    const uint32_t iw, const uint32_t ih, const uint32_t ow, const uint32_t oh, const uint32_t pw, const uint32_t ph,
    const uint32_t pad, const uint32_t pad_mode,
    const uint32_t stride_x, const uint32_t stride_y, const uint32_t dil_x, const uint32_t dil_y,
    const uint32_t act, const f32_t leaky_relu_coeff, const f32_t mask_coeff, const uint32_t seed
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
    const uint32_t oc_idx = nc_idx % channels;
    const uint32_t batch_idx = nc_idx / channels;

    extern __shared__ char shared_mem[];
    T *shared_input_tile = reinterpret_cast<T*>(shared_mem);
    const uint32_t tile_h = (bdim_y - 1) * stride_y + (ph - 1) * dil_y + 1;
    const uint32_t tile_w = (bdim_x - 1) * stride_x + (pw - 1) * dil_x + 1;

    // Prevent bank conflict
    const uint32_t tile_stride = tile_w % 32 == 0 ? tile_w + 1 : tile_w;
    const uint32_t total_tile_elems = tile_stride * tile_h;

    const int32_t in_h_start = static_cast<int32_t>(blk_y * bdim_y * stride_y) - static_cast<int32_t>(pad);
    const int32_t in_w_start = static_cast<int32_t>(blk_x * bdim_x * stride_x) - static_cast<int32_t>(pad);

    const uint32_t threads_per_block = bdim_x * bdim_y;
    const uint32_t address_offset = (batch_idx * channels + oc_idx) * (ih * iw);

    const T *in_ptr = preact_features + address_offset;

    // Read all the relevant input
    dev_conv_read_to_shared_memory<T>(
        shared_input_tile, in_ptr, tid, total_tile_elems, threads_per_block, tile_stride,
        tile_w, in_h_start, in_w_start, ih, iw, pad, pad_mode, act, leaky_relu_coeff,
        (pool_mode == 0 ? predrop_features : prepooling_features) + address_offset
    );

    // Ensure inputs are all read
    __syncthreads();

    f32_t out_val;

    if (ow_idx < ow && oh_idx < oh) {
        if (pool_mode == 0) {
            out_val = static_cast<f32_t>(shared_input_tile[ty * tile_stride + tx]);
        } else {
            const uint32_t local_h_start = ty * stride_y;
            const uint32_t local_w_start = tx * stride_x;

            f32_t acc_val = pool_mode == 1 ? static_cast<f32_t>(shared_input_tile[local_h_start * tile_stride + local_w_start])
                                           : 0.0f;

            #pragma unroll 4
            for (uint32_t ph_idx = 0; ph_idx < ph; ++ph_idx) {
                const uint32_t tile_row_offset = (local_h_start + ph_idx * dil_y) * tile_stride + local_w_start;

                for (uint32_t pw_idx = 0; pw_idx < pw; ++pw_idx) {
                    switch (pool_mode) {
                        case 2: // Sum Pooling
                        case 3: // Average Pooling
                            acc_val += static_cast<f32_t>(shared_input_tile[tile_row_offset + pw_idx * dil_x]);
                            break;
                        default: // 1: Max Pooling
                            acc_val = CudaMath<f32_t>::max(acc_val, static_cast<f32_t>(shared_input_tile[tile_row_offset + pw_idx * dil_x]));
                            break;
                    }
                }
            }

            if (pool_mode == 3) {
                acc_val /= static_cast<f32_t>(pw * ph);
            }

            out_val = acc_val;
        }

        const uint32_t idx = dev_nchw_idx(channels, oh, ow, batch_idx, oc_idx, oh_idx, ow_idx);

        if (pool_mode != 0) {
            predrop_features[idx] = static_cast<T>(out_val);
        }

        if (use_dropout) {
            if (dev_gen_random_f32_t(oh_idx, ow_idx, seed + oc_idx * 7) >= mask_coeff) {
                const f32_t scale = 1.0f / (1.0f - mask_coeff);
                mask[idx] = static_cast<T>(scale);
                features[idx] = static_cast<T>(out_val * scale);
            } else {
                mask[idx] = static_cast<T>(0.0f);
                features[idx] = static_cast<T>(0.0f);
            }
        } else {
            features[idx] = static_cast<T>(out_val);
        }
    }
}

#endif
