#pragma once

#ifndef BACKWARD_WMMA_CU
#define BACKWARD_WMMA_CU

#include "../math.cu"
#include "../util.cu"
#include <mma.h>
using namespace nvcuda::wmma;

#if __CUDA_ARCH__ >= 700

__device__ inline void compute_hidden_layer_error_wmma_kernel(
    const f32_t* next_d_prenorm_out, const f32_t* master_w_next, const f32_t* master_norm_w,
    f32_t* dx_out, f32_t* d_prenorm_out, f32_t* dNorm_w, f32_t* dNorm_b,
    const f16_t* norm_rstd, const f16_t* centered_out, const f16_t* prenorm_out,
    const f16_t* predrop_out, const f16_t* preact_out, const f16_t* mask,
    const uint32_t m, const uint32_t n, const uint32_t ec,
    const uint32_t norm, const uint32_t act, const f32_t leaky_relu_coeff
) {
    constexpr uint32_t TILE_DIM = 16;

    extern __shared__ f16_t back_hid_err_wmma_shared_mem[];
    f16_t* tile_A = back_hid_err_wmma_shared_mem;
    f16_t* tile_B = tile_A + TILE_DIM * TILE_DIM;

    const uint32_t block_row = blockIdx.y * TILE_DIM;
    const uint32_t block_col = blockIdx.x * TILE_DIM;
    const uint32_t thread_row = threadIdx.y;
    const uint32_t thread_col = threadIdx.x;
    const uint32_t tid = thread_row * TILE_DIM + thread_col;

    fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, f16_t, row_major> a_frag;
    fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, f16_t, col_major> b_frag;
    fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, f32_t> c_frag;

    fill_fragment(c_frag, 0.0f);

    // Load tile A and tile B
    for (uint32_t t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; ++t) {
        const uint32_t tile_offset = t * TILE_DIM;

        const uint32_t global_row_A = block_row + thread_row;
        if (const uint32_t global_col_A = tile_offset + thread_col; global_row_A < m && global_col_A < ec) {
            tile_A[tid] = static_cast<f16_t>(next_d_prenorm_out[global_row_A * ec + global_col_A]);
        } else {
            tile_A[tid] = static_cast<f16_t>(0.0f);
        }

        const uint32_t global_row_B = block_col + thread_row;
        if (const uint32_t global_col_B = tile_offset + thread_col; global_row_B < n && global_col_B < ec) {
            tile_B[tid] = static_cast<f16_t>(master_w_next[global_row_B * ec + global_col_B]);
        } else {
            tile_B[tid] = static_cast<f16_t>(0.0f);
        }

        __syncthreads();

        load_matrix_sync(a_frag, tile_A, TILE_DIM);
        load_matrix_sync(b_frag, tile_B, TILE_DIM);
        mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }


    const auto tile_C = reinterpret_cast<f32_t*>(back_hid_err_wmma_shared_mem);
    store_matrix_sync(tile_C, c_frag, TILE_DIM, mem_row_major);

    const uint32_t final_row = block_row + thread_row;
    if (const uint32_t final_col = block_col + thread_col; final_row < m && final_col < n) {
        const uint32_t idx = final_row * n + final_col;
        const f32_t next_layer_sum = tile_C[thread_row * TILE_DIM + thread_col];

        const f32_t inv_m = 1.0f / static_cast<f32_t>(m);
        const f32_t inv_n = 1.0f / static_cast<f32_t>(n);

        const f32_t derivative = dev_activation_derivative(
            act,
            next_layer_sum,
            predrop_out[idx],
            preact_out[idx],
            leaky_relu_coeff
        );

        // Calculate accumulated gradients
        const f32_t final_delta_f32 = static_cast<f32_t>(mask[idx]) * derivative;
        dx_out[idx] = final_delta_f32;

        dev_write_norm_gradients<f16_t>(
            d_prenorm_out, dNorm_w, dNorm_b,
            norm_rstd, centered_out, prenorm_out,
            final_delta_f32, master_norm_w[final_col], 1.0f,
            norm, idx, inv_n, inv_m
        );
    }
}

#endif

#endif