#pragma once

#ifndef FORWARD_WMMA_CU
#define FORWARD_WMMA_CU

#include "../math.cu"
#include <mma.h>
using namespace nvcuda::wmma;

#if __CUDA_ARCH__ >= 700
__device__ inline void forward_pass_0_wmma_kernel(
    f16_t* prenorm_out, const f16_t* in, const f16_t* w, const f16_t* b,
    const uint32_t use_bias, const uint32_t m, const uint32_t n, const uint32_t wc
) {
    constexpr uint32_t TILE_DIM = 16;

    extern __shared__ f16_t fr_wmma_shared_mem[];
    f16_t* tile_A = fr_wmma_shared_mem;
    f16_t* tile_B = tile_A + TILE_DIM * TILE_DIM;

    const uint32_t block_row = blockIdx.y * TILE_DIM;
    const uint32_t block_col = blockIdx.x * TILE_DIM;
    const uint32_t thread_row = threadIdx.y;
    const uint32_t thread_col = threadIdx.x;
    const uint32_t tid = thread_row * TILE_DIM + thread_col;

    fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, f16_t, row_major> a_frag;
    fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, f16_t, row_major> b_frag;
    fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, f32_t> c_frag;

    fill_fragment(c_frag, 0.0f);

    // Load tile A and tile B
    for (uint32_t t = 0; t < (n + TILE_DIM - 1) / TILE_DIM; ++t) {
        const uint32_t tile_offset = t * TILE_DIM;

        const uint32_t global_row_A = block_row + thread_row;
        if (const uint32_t global_col_A = tile_offset + thread_col; global_row_A < m && global_col_A < n) {
            tile_A[tid] = in[global_row_A * n + global_col_A];
        } else {
            tile_A[tid] = static_cast<f16_t>(0.0f);
        }

        const uint32_t global_row_B = tile_offset + thread_row;
        if (const uint32_t global_col_B = block_col + thread_col; global_row_B < n && global_col_B < wc) {
            tile_B[tid] = w[global_row_B * wc + global_col_B];
        } else {
            tile_B[tid] = static_cast<f16_t>(0.0f);
        }

        __syncthreads();

        load_matrix_sync(a_frag, tile_A, TILE_DIM);
        load_matrix_sync(b_frag, tile_B, TILE_DIM);
        mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    const auto tile_C = reinterpret_cast<f32_t*>(fr_wmma_shared_mem);
    store_matrix_sync(tile_C, c_frag, TILE_DIM, mem_row_major);

    const uint32_t final_row = block_row + thread_row;
    if (const uint32_t final_col = block_col + thread_col; final_row < m && final_col < wc) {
        f32_t out_val = tile_C[thread_row * TILE_DIM + thread_col];

        if (use_bias) {
            out_val += static_cast<f32_t>(b[final_col]);
        }

        prenorm_out[final_row * wc + final_col] = static_cast<f16_t>(out_val);
    }
}
#endif

#endif