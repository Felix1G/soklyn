#pragma once

#include "kernel/types.hpp"
#include "kernel/math.cu"
#include "kernel/util.cu"
#include "kernel/ffn/backward.cu"
#include "kernel/ffn/backward_wmma.cu"
#include "kernel/ffn/forward.cu"
#include "kernel/ffn/forward_wmma.cu"
#include "kernel/conv/forward.cu"

extern "C" {

__global__ void cast_f32_f16_t_kernel(const f32_t* in, f16_t* out, const uint32_t len) {
    cast_f32_f16_t(in, out, len);
}

__global__ void cast_f16_f32_t_kernel(const f16_t* in, f32_t* out, const uint32_t len) {
    cast_f16_f32_t(in, out, len);
}

__global__ void broadcast_kernel_f32(f32_t* dst, const f32_t v, const uint32_t len) {
    broadcast_kernel<f32_t>(dst, v, len);
}

__global__ void broadcast_kernel_f16(f16_t* dst, const f16_t v, const uint32_t len) {
    broadcast_kernel<f16_t>(dst, v, len);
}

__global__ void sgemm_kernel(const f32_t* a, const f32_t* b, f32_t* c, const uint32_t m, const uint32_t n, const uint32_t p, const uint32_t tile_dim) {
    gemm_kernel<f32_t>(a, b, c, m, n, p, tile_dim);
}

__global__ void hgemm_kernel(const f16_t* a, const f16_t* b, f16_t* c, const uint32_t m, const uint32_t n, const uint32_t p, const uint32_t tile_dim) {
    gemm_kernel<f16_t>(a, b, c, m, n, p, tile_dim);
}

__global__ void sgeam_kernel(const f32_t* a, const f32_t* b, f32_t* c, const uint32_t m, const uint32_t n) {
    geam_kernel<f32_t>(a, b, c, m, n);
}

__global__ void hgeam_kernel(const f16_t* a, const f16_t* b, f16_t* c, const uint32_t m, const uint32_t n) {
    geam_kernel<f16_t>(a, b, c, m, n);
}

__global__ void softmax_kernel_f32(f32_t* out, const uint32_t m, const uint32_t n) {
    softmax_kernel<f32_t>(out, m, n);
}

__global__ void softmax_kernel_f16(f16_t* out, const uint32_t m, const uint32_t n) {
    softmax_kernel<f16_t>(out, m, n);
}

__global__ void forward_pass_0_f32(
    f32_t* prenorm_out, const f32_t* in, const f32_t* w, const f32_t* b,
    const uint32_t use_bias, const uint32_t m, const uint32_t n, const uint32_t wc, const uint32_t tile_dim
) {
    forward_pass_0_kernel<f32_t>(prenorm_out, in, w, b, use_bias, m, n, wc, tile_dim);
}

__global__ void forward_pass_0_f16(
    f16_t* prenorm_out, const f16_t* in, const f16_t* w, const f16_t* b,
    const uint32_t use_bias, const uint32_t m, const uint32_t n, const uint32_t wc, const uint32_t tile_dim
) {
#if __CUDA_ARCH__ >= 700
    forward_pass_0_wmma_kernel(prenorm_out, in, w, b, use_bias, m, n, wc);
#else
    forward_pass_0_kernel<f16_t>(prenorm_out, in, w, b, use_bias, m, n, wc, tile_dim);
#endif
}

__global__ void forward_pass_1_f32(
    f32_t* preact_out, f32_t* centered_out, const f32_t* prenorm_out,
    const f32_t* norm_w, const f32_t* norm_b, f32_t* norm_rstd,
    const uint32_t m, const uint32_t wc, const uint32_t norm
) {
    forward_pass_1_kernel<f32_t>(preact_out, centered_out, prenorm_out, norm_w, norm_b, norm_rstd, m, wc, norm);
}

__global__ void forward_pass_1_f16(
    f16_t* preact_out, f16_t* centered_out, const f16_t* prenorm_out,
    const f16_t* norm_w, const f16_t* norm_b, f16_t* norm_rstd,
    const uint32_t m, const uint32_t wc, const uint32_t norm
) {
    forward_pass_1_kernel<f16_t>(preact_out, centered_out, prenorm_out, norm_w, norm_b, norm_rstd, m, wc, norm);
}

__global__ void forward_pass_2_f32(
    f32_t* out, f32_t* predrop_out, const f32_t* preact_out, f32_t* mask,
    const uint32_t m, const uint32_t n, const uint32_t act,
    const f32_t leaky_relu_coeff, const uint32_t use_dropout, const f32_t mask_coeff, const uint32_t seed
) {
    forward_pass_2_kernel<f32_t>(out, predrop_out, preact_out, mask, m, n, act, leaky_relu_coeff, use_dropout,
                                 mask_coeff, seed);
}

__global__ void forward_pass_2_f16(
    f16_t* out, f16_t* predrop_out, const f16_t* preact_out, f16_t* mask,
    const uint32_t m, const uint32_t n, const uint32_t act,
    const f32_t leaky_relu_coeff, const uint32_t use_dropout, const f32_t mask_coeff, const uint32_t seed
) {
    forward_pass_2_kernel<f16_t>(out, predrop_out, preact_out, mask, m, n, act, leaky_relu_coeff, use_dropout,
                                 mask_coeff, seed);
}

__global__ void compute_output_layer_error_f32(
    const f32_t* out, const f32_t* preact_out, const f32_t* target, const f32_t* master_norm_w,
    f32_t* dx_out, f32_t* d_prenorm_out, f32_t* dNorm_w, f32_t* dNorm_b,
    const f32_t* norm_rstd, const f32_t* centered_out, const f32_t* prenorm_out,
    const uint32_t m, const uint32_t n, const uint32_t err_mode,
    const uint32_t norm, const uint32_t act, const f32_t leaky_relu_coeff
) {
    compute_output_layer_error_kernel<f32_t>(
        out, preact_out, target, master_norm_w, dx_out, d_prenorm_out, dNorm_w, dNorm_b, norm_rstd, centered_out,
        prenorm_out, m, n, err_mode, norm, act, leaky_relu_coeff
    );
}

__global__ void compute_output_layer_error_f16(
    const f16_t* out, const f16_t* preact_out, const f16_t* target, const f32_t* master_norm_w,
    f32_t* dx_out, f32_t* d_prenorm_out, f32_t* dNorm_w, f32_t* dNorm_b,
    const f16_t* norm_rstd, const f16_t* centered_out, const f16_t* prenorm_out,
    const uint32_t m, const uint32_t n, const uint32_t err_mode,
    const uint32_t norm, const uint32_t act, const f32_t leaky_relu_coeff
) {
    compute_output_layer_error_kernel<f16_t>(
        out, preact_out, target, master_norm_w, dx_out, d_prenorm_out, dNorm_w, dNorm_b, norm_rstd, centered_out,
        prenorm_out, m, n, err_mode, norm, act, leaky_relu_coeff
    );
}

__global__ void compute_hidden_layer_error_f32(
    const f32_t* next_d_prenorm_out, const f32_t* master_w_next, const f32_t* master_norm_w,
    f32_t* dx_out, f32_t* d_prenorm_out, f32_t* dNorm_w, f32_t* dNorm_b,
    const f32_t* norm_rstd, const f32_t* centered_out, const f32_t* prenorm_out,
    const f32_t* predrop_out, const f32_t* preact_out, const f32_t* mask,
    const uint32_t m, const uint32_t n, const uint32_t ec,
    const uint32_t norm, const uint32_t act, const f32_t leaky_relu_coeff
) {
    compute_hidden_layer_error_kernel<f32_t>(
        next_d_prenorm_out, master_w_next, master_norm_w, dx_out, d_prenorm_out, dNorm_w, dNorm_b, norm_rstd,
        centered_out, prenorm_out, predrop_out, preact_out, mask, m, n, ec, norm, act, leaky_relu_coeff
    );
}

__global__ void compute_hidden_layer_error_f16(
    const f32_t* next_d_prenorm_out, const f32_t* master_w_next, const f32_t* master_norm_w,
    f32_t* dx_out, f32_t* d_prenorm_out, f32_t* dNorm_w, f32_t* dNorm_b,
    const f16_t* norm_rstd, const f16_t* centered_out, const f16_t* prenorm_out,
    const f16_t* predrop_out, const f16_t* preact_out, const f16_t* mask,
    const uint32_t m, const uint32_t n, const uint32_t ec,
    const uint32_t norm, const uint32_t act, const f32_t leaky_relu_coeff
) {
#if __CUDA_ARCH__ >= 700
    compute_hidden_layer_error_wmma_kernel(
        next_d_prenorm_out, master_w_next, master_norm_w, dx_out, d_prenorm_out, dNorm_w, dNorm_b, norm_rstd,
        centered_out, prenorm_out, predrop_out, preact_out, mask, m, n, ec, norm, act, leaky_relu_coeff
    );
#else
    compute_hidden_layer_error_kernel<f16_t>(
        next_d_prenorm_out, w_next, norm_w, dx_out, d_prenorm_out, dNorm_w, dNorm_b, norm_rstd, centered_out,
        prenorm_out, predrop_out, preact_out, mask, m, n, ec, norm, act, leaky_relu_coeff
    );
#endif
}

__global__ void backward_pass_f32(
    const f32_t* x, f32_t* w, f32_t* b, f32_t* master_w, f32_t* master_b, f32_t* d_prenorm_out,
    f32_t* dv_w, f32_t* dv_b, f32_t* dm_w, f32_t* dm_b,
    f32_t* norm_w, f32_t* norm_b, f32_t* master_norm_w, f32_t* master_norm_b, f32_t* dNorm_w, f32_t* dNorm_b,
    f32_t* dv_norm_w, f32_t* dv_norm_b, f32_t* dm_norm_w, f32_t* dm_norm_b,
    const uint32_t use_bias, const uint32_t norm,
    const uint32_t m, const uint32_t n, const uint32_t wr, const f32_t lr, const f32_t max_grad_norm,
    const uint32_t optimiser, const uint32_t norm_optimiser,
    const f32_t linear_beta1, const f32_t linear_beta2, const f32_t linear_epsilon, const uint32_t linear_nesterov,
    const f32_t norm_beta1, const f32_t norm_beta2, const f32_t norm_epsilon, const uint32_t norm_nesterov,
    const uint32_t regularisation, const f32_t regu_coeff, const uint32_t step
) {
    backward_pass_kernel<f32_t>(
        x, w, b, master_w, master_b, d_prenorm_out, dv_w, dv_b, dm_w, dm_b,
        norm_w, norm_b, master_norm_w, master_norm_b, dNorm_w, dNorm_b,
        dv_norm_w, dv_norm_b, dm_norm_w, dm_norm_b,
        use_bias, norm, m, n, wr, lr, max_grad_norm,
        optimiser, norm_optimiser,
        linear_beta1, linear_beta2, linear_epsilon, linear_nesterov,
        norm_beta1, norm_beta2, norm_epsilon, norm_nesterov,
        regularisation, regu_coeff, step
    );
}

__global__ void backward_pass_f16(
    const f16_t* x, f16_t* w, f16_t* b, f32_t* master_w, f32_t* master_b, f32_t* d_prenorm_out,
    f32_t* dv_w, f32_t* dv_b, f32_t* dm_w, f32_t* dm_b,
    f16_t* norm_w, f16_t* norm_b, f32_t* master_norm_w, f32_t* master_norm_b, f32_t* dNorm_w, f32_t* dNorm_b,
    f32_t* dv_norm_w, f32_t* dv_norm_b, f32_t* dm_norm_w, f32_t* dm_norm_b,
    const uint32_t use_bias, const uint32_t norm,
    const uint32_t m, const uint32_t n, const uint32_t wr, const f32_t lr, const f32_t max_grad_norm,
    const uint32_t optimiser, const uint32_t norm_optimiser,
    const f32_t linear_beta1, const f32_t linear_beta2, const f32_t linear_epsilon, const uint32_t linear_nesterov,
    const f32_t norm_beta1, const f32_t norm_beta2, const f32_t norm_epsilon, const uint32_t norm_nesterov,
    const uint32_t regularisation, const f32_t regu_coeff, const uint32_t step
) {
    backward_pass_kernel<f16_t>(
        x, w, b, master_w, master_b, d_prenorm_out, dv_w, dv_b, dm_w, dm_b,
        norm_w, norm_b, master_norm_w, master_norm_b, dNorm_w, dNorm_b,
        dv_norm_w, dv_norm_b, dm_norm_w, dm_norm_b,
        use_bias, norm, m, n, wr, lr, max_grad_norm,
        optimiser, norm_optimiser,
        linear_beta1, linear_beta2, linear_epsilon, linear_nesterov,
        norm_beta1, norm_beta2, norm_epsilon, norm_nesterov,
        regularisation, regu_coeff, step
    );
}

__global__ void conv_forward_pass_0_kernel_f32(
    f32_t* prenorm_features, const f32_t* in, const f32_t* w, const f32_t* b,
    const uint32_t use_bias, const uint32_t pad_mode, const uint32_t ic, const uint32_t oc,
    const uint32_t iw, const uint32_t ih, const uint32_t ow, const uint32_t oh, const uint32_t fw, const uint32_t fh,
    const uint32_t pad, const uint32_t stride_x, const uint32_t stride_y, const uint32_t dil_x, const uint32_t dil_y
) {
    conv_forward_pass_0_kernel<f32_t>(
        prenorm_features, in, w, b, use_bias, pad_mode, ic, oc, iw, ih, ow, oh, fw, fh,
        pad, stride_x, stride_y, dil_x, dil_y
    );
}

__global__ void conv_forward_pass_0_kernel_f16(
    f16_t* prenorm_features, const f16_t* in, const f16_t* w, const f16_t* b,
    const uint32_t use_bias, const uint32_t pad_mode, const uint32_t ic, const uint32_t oc,
    const uint32_t iw, const uint32_t ih, const uint32_t ow, const uint32_t oh, const uint32_t fw, const uint32_t fh,
    const uint32_t pad, const uint32_t stride_x, const uint32_t stride_y, const uint32_t dil_x, const uint32_t dil_y
) {
    conv_forward_pass_0_kernel<f16_t>(
        prenorm_features, in, w, b, use_bias, pad_mode, ic, oc, iw, ih, ow, oh, fw, fh,
        pad, stride_x, stride_y, dil_x, dil_y
    );
}

__global__ void conv_forward_pass_1_kernel_f32(
    f32_t* __restrict__ preact_features, f32_t* __restrict__ centered_features, const f32_t* __restrict__ prenorm_features,
    const f32_t* __restrict__ norm_w, const f32_t* __restrict__ norm_b, f32_t* __restrict__ norm_rstd,
    const uint32_t ow, const uint32_t oh, const uint32_t oc, const uint32_t on, const uint32_t norm
) {
    conv_forward_pass_1_kernel<f32_t>(
        preact_features, centered_features, prenorm_features, norm_w, norm_b, norm_rstd, ow, oh, oc, on, norm
    );
}

__global__ void conv_forward_pass_1_kernel_f16(
    f16_t* __restrict__ preact_features, f16_t* __restrict__ centered_features, const f16_t* __restrict__ prenorm_features,
    const f16_t* __restrict__ norm_w, const f16_t* __restrict__ norm_b, f16_t* __restrict__ norm_rstd,
    const uint32_t ow, const uint32_t oh, const uint32_t oc, const uint32_t on, const uint32_t norm
) {
    conv_forward_pass_1_kernel<f16_t>(
        preact_features, centered_features, prenorm_features, norm_w, norm_b, norm_rstd, ow, oh, oc, on, norm
    );
}

}