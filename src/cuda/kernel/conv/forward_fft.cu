#pragma once

#ifndef CONV_FORWARD_FFT_CU
#define CONV_FORWARD_FFT_CU

#include "forward.cuh"
#include <math_constants.h>

__device__ __forceinline__ uint32_t dev_bit_reverse(const uint32_t idx, const uint32_t log2_N) {
    return __brev(idx) >> (32 - log2_N);
}

template<bool INVERSE>
__device__ __forceinline__ void dev_cooley_tukey_1d_fft(
    cuFloatComplex *shared_mem,
    const cuFloatComplex *twiddle_lut,
    const uint32_t N,
    const uint32_t thread,
    const uint32_t block_dim
) {
    for (uint32_t group_size = 2; group_size <= N; group_size <<= 1) {
        const uint32_t half_group_size = group_size >> 1;
        const uint32_t lut_stride = N / group_size;

        // Max 8 butterfly pairs per thread (handles up to N = 4096 with blockDim = 256  [2 x 8 x 256])
        constexpr uint32_t MAX_ITEMS = 8;
        cuFloatComplex res_u[MAX_ITEMS];
        cuFloatComplex res_v[MAX_ITEMS];

        uint32_t read_count = 0;

        // compute the result for this stage
        for (uint32_t idx = thread; idx < N >> 1; idx += block_dim) {
            const uint32_t group_idx = idx / half_group_size;
            const uint32_t twiddle_k = idx % half_group_size;
            const uint32_t u_idx = group_idx * group_size + twiddle_k;
            const uint32_t v_idx = u_idx + half_group_size;

            cuFloatComplex twiddle = __ldg(&twiddle_lut[twiddle_k * lut_stride]);
            if constexpr (INVERSE) {
                twiddle = cuConjf(twiddle);
            }

            const cuFloatComplex u = shared_mem[u_idx];
            const cuFloatComplex v = shared_mem[v_idx];
            const cuFloatComplex t = cuCmulf(v, twiddle);

            res_u[read_count] = cuCaddf(u, t);
            res_v[read_count] = cuCsubf(u, t);
            read_count++;
        }

        // wait until all values have been read
        __syncthreads();

        // write data back into the shared memory
        read_count = 0;
        for (uint32_t idx = thread; idx < N >> 1; idx += block_dim) {
            const uint32_t group_idx = idx / half_group_size;
            const uint32_t twiddle_k = idx % half_group_size;
            const uint32_t u_idx = group_idx * group_size + twiddle_k;
            const uint32_t v_idx = u_idx + half_group_size;

            shared_mem[u_idx] = res_u[read_count];
            shared_mem[v_idx] = res_v[read_count];
            read_count++;
        }

        // wait until all is done
        __syncthreads();
    }
}

template<typename T, bool IS_ROW_PASS>
__device__ __forceinline__ void dev_load_tensor_to_shared(
    cuFloatComplex *shared_fft_mem,
    const T *src_ptr,
    const uint32_t block,
    const uint32_t input_blocks,
    const uint32_t sweep_len, // oh for column, ow for row
    const uint32_t fixed_dim_size, // ow for column, oh for row
    const uint32_t ih, const uint32_t iw,
    const uint32_t fh, const uint32_t fw,
    const uint32_t dil_y, const uint32_t dil_x,
    const uint32_t pad, const uint32_t pad_mode,
    const uint32_t log2_N,
    const uint32_t thread, const uint32_t block_dim
) {
    const bool is_input = block < input_blocks;
    const uint32_t sub_block = is_input ? block : block - input_blocks;

    const int32_t fixed_coord = static_cast<int32_t>(sub_block % fixed_dim_size);
    const uint32_t kernel_idx = sub_block / fixed_dim_size;

    const uint32_t spatial_slice_size = is_input ? ih * iw : fh * fw;
    const T *base_ptr = src_ptr + kernel_idx * spatial_slice_size;

    const uint32_t actual_fw = (fw - 1) * dil_x + 1;
    const uint32_t actual_fh = (fh - 1) * dil_y + 1;
    const int32_t pad_i = static_cast<int32_t>(pad);

    for (uint32_t s_idx = thread; s_idx < sweep_len; s_idx += block_dim) {
        const uint32_t br_idx = dev_bit_reverse(s_idx, log2_N);
        cuFloatComplex fill_val = make_cuFloatComplex(0.0f, 0.0f);

        if (is_input) {
            int32_t row, col;
            if constexpr (IS_ROW_PASS) {
                row = fixed_coord - pad_i;
                col = static_cast<int32_t>(s_idx) - pad_i;
            } else {
                // column pass
                row = static_cast<int32_t>(s_idx) - pad_i;
                col = fixed_coord - pad_i;
            }

            // ======== INPUT TENSOR BRANCH ========
            const bool is_in_bounds = col >= 0 && col < static_cast<int32_t>(iw) &&
                                      row >= 0 && row < static_cast<int32_t>(ih);

            if (is_in_bounds) {
                const uint32_t in_idx = static_cast<uint32_t>(row) * iw + static_cast<uint32_t>(col);
                if constexpr (std::is_same_v<T, cuFloatComplex>) {
                    fill_val = base_ptr[in_idx];
                } else {
                    fill_val = make_cuFloatComplex(static_cast<f32_t>(base_ptr[in_idx]), 0.0f);
                }
            } else if (pad_mode != 0) {
                if constexpr (std::is_same_v<T, cuFloatComplex>) {
                    fill_val = make_cuFloatComplex(0.0, 0.0); // padding is not supported for complex numbers
                } else {
                    if (col < static_cast<int32_t>(iw) + pad_i && row < static_cast<int32_t>(ih) + pad_i) {
                        fill_val = make_cuFloatComplex(
                            static_cast<f32_t>(dev_conv_apply_padding(base_ptr, pad_mode, iw, ih, col, row)), 0.0f);
                    } else {
                        // outside actual inputs, this part is the padding only for the fft.
                        fill_val = make_cuFloatComplex(0.0, 0.0);
                    }
                }
            }
        } else {
            // ======== FILTER WEIGHTS BRANCH ========
            // Since only row fft loads the original filter anyway, then I won't need to do handle the column part :)
            if constexpr (IS_ROW_PASS) {
                const int32_t freq_map_col = static_cast<int32_t>(s_idx);

                const int32_t total_row    = static_cast<int32_t>(fixed_dim_size);
                const int32_t total_col    = static_cast<int32_t>(sweep_len);

                const int32_t start_row = (total_row - actual_fh) / 2;
                const int32_t start_col = (total_col - actual_fw) / 2;

                // fixed_coord is freq map row
                const int32_t filter_row = (fixed_coord + start_row) % total_row;
                const int32_t filter_col = (freq_map_col + start_col) % total_col;

                const bool is_in_bounds = filter_row < actual_fh && filter_row % dil_y == 0 &&
                                      filter_col < actual_fw && filter_col % dil_x == 0;

                if (is_in_bounds) {
                    const uint32_t native_row = static_cast<uint32_t>(filter_row) / dil_y;
                    const uint32_t native_col = static_cast<uint32_t>(filter_col) / dil_x;

                    const uint32_t w_idx = native_row * fw + native_col;

                    fill_val = make_cuFloatComplex(static_cast<f32_t>(base_ptr[w_idx]), 0.0f);
                }
            } else {
                fill_val = base_ptr[s_idx * fw + fixed_coord];
            }
        }

        shared_fft_mem[br_idx] = fill_val;
    }
}

// Each block computes a single row (samples lie horizontal). The block dimension is the tile dimension squared.
// For dilation > 1, the size of the row will also be 'dilated', hence the indices in between elements will be 0.
template<typename T>
__device__ void conv_fft_row_transform_kernel(
    cuFloatComplex * __restrict__ fft_in, cuFloatComplex * __restrict__ fft_w,
    const cuFloatComplex * __restrict__ twiddle_lut, const T * __restrict__ in, const T * __restrict__ w,
    const uint32_t batches, const uint32_t ic,
    const uint32_t iw, const uint32_t ih, const uint32_t ow, const uint32_t oh, const uint32_t fw, const uint32_t fh,
    const uint32_t pad, const uint32_t pad_mode, const uint32_t dil_x, const uint32_t dil_y
) {
    extern __shared__ cuFloatComplex shared_fft_mem[];

    const uint32_t thread = threadIdx.x;
    const uint32_t block = blockIdx.x; // batch * ic * oh + oc * ic * oh
    const uint32_t block_dim = blockDim.x;
    const uint32_t log2_N = 31 - __clz(ow);

    const uint32_t input_blocks = batches * ic * oh;

    // load and pad data into shared memory
    dev_load_tensor_to_shared<T, true>(
        shared_fft_mem,
        block < input_blocks ? in : w, // input vs filter pointer
        block, input_blocks, ow, oh, ih, iw, fh, fw, dil_y, dil_x,
        pad, pad_mode, log2_N, thread, block_dim
    );

    // ensure all shared memory is saved
    __syncthreads();

    // apply the FFT
    dev_cooley_tukey_1d_fft<false>(shared_fft_mem, twiddle_lut, ow, thread, block_dim);

    // store transformed spectrum to global memory
    const uint32_t image_size = oh * ow;
    const uint32_t ptr_offset = block < input_blocks
                                    ? block / oh * image_size
                                    : (block - input_blocks) / oh * image_size;

    cuFloatComplex *fft_ptr = block < input_blocks
                                  ? fft_in + ptr_offset
                                  : fft_w + ptr_offset;
    const uint32_t y_idx = block % oh;
    for (uint32_t x_idx = thread; x_idx < ow; x_idx += block_dim) {
        fft_ptr[y_idx * ow + x_idx] = shared_fft_mem[x_idx];
    }
}

// Each block computes a single column (samples lie vertical). The block dimension is the tile dimension squared.
__global__ void conv_fft_col_transform_kernel(
    cuFloatComplex * __restrict__ fft_in, cuFloatComplex * __restrict__ fft_w,
    const cuFloatComplex * __restrict__ twiddle_lut,
    const uint32_t batches, const uint32_t ic, const uint32_t ow, const uint32_t oh
) {
    extern __shared__ cuFloatComplex shared_fft_mem[];

    const uint32_t thread = threadIdx.x;
    const uint32_t block = blockIdx.x; // batch * ic * ow + oc * ic * ow
    const uint32_t block_dim = blockDim.x;
    const uint32_t log2_N = 31 - __clz(oh);

    const uint32_t input_blocks = batches * ic * ow;

    // load and pad data into shared memory
    dev_load_tensor_to_shared<cuFloatComplex, false>(
        shared_fft_mem,
        block < input_blocks ? fft_in : fft_w,
        block, input_blocks, oh, ow, oh, ow, oh, ow,
        1, 1, 0, 0, log2_N, thread, block_dim
    );

    // ensure all shared memory is saved
    __syncthreads();

    // apply the FFT
    dev_cooley_tukey_1d_fft<false>(shared_fft_mem, twiddle_lut, oh, thread, block_dim);

    // store transformed spectrum to global memory
    const uint32_t image_size = oh * ow;
    const uint32_t ptr_offset = block < input_blocks
                                    ? block / ow * image_size
                                    : (block - input_blocks) / ow * image_size;

    cuFloatComplex *fft_ptr = block < input_blocks
                                  ? fft_in + ptr_offset
                                  : fft_w + ptr_offset;
    const uint32_t x_idx = block % ow;
    for (uint32_t y_idx = thread; y_idx < oh; y_idx += block_dim) {
        fft_ptr[y_idx * ow + x_idx] = shared_fft_mem[y_idx];
    }
}

// Each block computes a single row (samples lie horizontal). The block dimension is the tile dimension squared.
__global__ void conv_elem_mul_ifft_row_kernel(
    const cuFloatComplex* __restrict__ fft_in, const cuFloatComplex* __restrict__ fft_w,
    cuFloatComplex* __restrict__ fft_out,
    const cuFloatComplex* __restrict__ twiddle_lut,
    const uint32_t oc, const uint32_t ic, const uint32_t ow, const uint32_t oh
) {
    extern __shared__ cuFloatComplex shared_fft_mem[];

    const uint32_t thread = threadIdx.x;
    const uint32_t block = blockIdx.x; // batch * oc * oh
    const uint32_t block_dim = blockDim.x;
    const uint32_t log2_N = 31 - __clz(ow);

    const uint32_t row_idx = block % oh;
    const uint32_t kernel_idx = block / oh;
    const uint32_t oc_idx = kernel_idx % oc;
    const uint32_t batch_idx = kernel_idx / oc;

    // load values into shared memory
    for (uint32_t s_idx = thread; s_idx < ow; s_idx += block_dim) {
        cuFloatComplex val = make_cuFloatComplex(0.0f, 0.0f);

        // element multiplication
        for (uint32_t ic_idx = 0; ic_idx < ic; ++ic_idx) {
            const uint32_t w_idx = dev_nchw_idx(ic, oh, ow, oc_idx, ic_idx, row_idx, s_idx);
            const uint32_t in_idx = dev_nchw_idx(ic, oh, ow, batch_idx, ic_idx, row_idx, s_idx);

            val = cuCaddf(val, cuCmulf(fft_in[in_idx], fft_w[w_idx]));
        }

        // save to shared memory
        const uint32_t br_idx = dev_bit_reverse(s_idx, log2_N);
        shared_fft_mem[br_idx] = val;
    }

    // ensure all shared memory is saved
    __syncthreads();

    // apply the FFT
    dev_cooley_tukey_1d_fft<true>(shared_fft_mem, twiddle_lut, ow, thread, block_dim);

    // store transformed spectrum to global memory
    cuFloatComplex *fft_ptr = fft_out + block * ow;
    for (uint32_t x_idx = thread; x_idx < ow; x_idx += block_dim) {
        fft_ptr[x_idx] = shared_fft_mem[x_idx];
    }
}

// Each block computes a single column (samples lie vertical).
// Parts of the frequency map that are not part of the output (due to stride) are skipped.
// The number of blocks is based on this fact.
//
// The block dimension is the tile dimension squared.
template<typename T>
__device__ void conv_ifft_col_transform_kernel(
    T * __restrict__ prenorm_features, cuFloatComplex* __restrict__ fft_out,
    const cuFloatComplex* __restrict__ twiddle_lut, const T * __restrict__ b,
    const uint32_t use_bias, const uint32_t oc,
    const uint32_t ow, const uint32_t oh, const uint32_t out_w, const uint32_t out_h,
    const uint32_t stride_x, const uint32_t stride_y
) {
    extern __shared__ cuFloatComplex shared_fft_mem[];

    const uint32_t thread = threadIdx.x;
    const uint32_t block = blockIdx.x; // batch * oc * out_w <- skipping unread columns
    const uint32_t block_dim = blockDim.x;
    const uint32_t log2_N = 31 - __clz(oh);

    const uint32_t feature_col_idx = block % out_w;
    const uint32_t col_idx = block % out_w * stride_x;
    const uint32_t kernel_idx = block / out_w;
    const uint32_t oc_idx = kernel_idx % oc;

    // load values into shared memory
    const cuFloatComplex* fft_ptr = fft_out + kernel_idx * oh * ow;
    for (uint32_t s_idx = thread; s_idx < oh; s_idx += block_dim) {
        const uint32_t br_idx = dev_bit_reverse(s_idx, log2_N);
        shared_fft_mem[br_idx] = fft_ptr[s_idx * ow + col_idx];
    }

    // ensure all shared memory is saved
    __syncthreads();

    // apply the FFT
    dev_cooley_tukey_1d_fft<true>(shared_fft_mem, twiddle_lut, oh, thread, block_dim);

    // store transformed spectrum to global memory
    const T bias = use_bias ? b[oc_idx] : static_cast<T>(0.0f);
    const f32_t norm_factor = 1.0f / (ow * oh);

    T *features = prenorm_features + kernel_idx * out_h * out_w;

    for (uint32_t y_idx = thread; y_idx < out_h; y_idx += block_dim) {
        const T value = static_cast<T>(cuCrealf(shared_fft_mem[y_idx * stride_y]) * norm_factor);
        features[y_idx * out_w + feature_col_idx] = value + bias;
    }
}

#endif
