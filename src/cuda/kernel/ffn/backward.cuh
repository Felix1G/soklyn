#pragma once

#ifndef FFN_BACKWARD_CUH
#define FFN_BACKWARD_CUH

#include "../util.cuh"
#include "../math.cuh"

/**
 * Computes the error deltas for the output layer.
 * @tparam T Either f32_t or f16_t.
 * @param out The output matrix (serving as input for the backward pass).
 * @param preact_out The matrix representing the output before activation.
 * @param target The target matrix (containing the expected values).
 * @param master_norm_w The master normalisation weights matrix of the current layer.
 * @param dx_out The error deltas matrix of the current layer.
 * @param d_prenorm_out The accumulated delta prenorm out matrix of this layer.
 * @param dNorm_w The accumulated delta norm weights matrix of this layer.
 * @param dNorm_b The accumulated delta norm biases matrix of this layer.
 * @param norm_rstd The matrix representing the reciprocal of standard deviation.
 * @param centered_out The centered values matrix (from normalisation).
 * @param prenorm_out The matrix representing the output before normalisation.
 * @param m The rows of the output matrix (batch size).
 * @param n The columns of the output matrix (output features).
 * @param err_mode The mode of the loss function.
 * @param norm The normalisation mode used.
 * @param act The activation mode for this output layer.
 * @param leaky_relu_coeff A coefficient specially for the LeakyReLU activation function as a multiplier for negative values.
 */
template<typename T> __device__ void compute_output_layer_error_kernel(
    const T* __restrict__ out, const T* __restrict__ preact_out, const T* __restrict__ target, const f32_t* master_norm_w,
    f32_t* dx_out, f32_t* d_prenorm_out, f32_t* dNorm_w, f32_t* dNorm_b,
    const T* __restrict__ norm_rstd, const T* __restrict__ centered_out, const T* __restrict__ prenorm_out,
    uint32_t m, uint32_t n, uint32_t err_mode,
    uint32_t norm, uint32_t act, f32_t leaky_relu_coeff
);

/**
 *
 * @tparam T Either f32_t or f16_t.
 * @param next_d_prenorm_out The error deltas after passing the normalisation derivative matrix from the next layer.
 * @param master_w_next The master weights matrix from the next layer.
 * @param master_norm_w The master normalisation weights matrix of the current layer.
 * @param dx_out The error deltas matrix of the current layer.
 * @param d_prenorm_out The accumulated delta prenorm out matrix of this layer.
 * @param dNorm_w The accumulated delta norm weights matrix of this layer.
 * @param dNorm_b The accumulated delta norm biases matrix of this layer.
 * @param norm_rstd The matrix representing the reciprocal of standard deviation.
 * @param centered_out The centered values matrix (from normalisation).
 * @param prenorm_out The matrix representing the output before normalisation.
 * @param predrop_out The matrix representing the output before dropout.
 * @param preact_out The matrix representing the output before activation.
 * @param mask The masking matrix.
 * @param m The rows of the current layer's error deltas matrix (batch size).
 * @param n The columns of the current layer's error deltas matrix (output features).
 * @param ec The columns of the next layer's error deltas matrix (next layer's output features).
 * @param norm The normalisation mode used.
 * @param act The activation mode used.
 * @param leaky_relu_coeff A coefficient specially for the LeakyReLU activation function as a multiplier for negative values.
 */
template<typename T> __device__ void compute_hidden_layer_error_kernel(
    const f32_t* next_d_prenorm_out, const f32_t* master_w_next, const f32_t* master_norm_w,
    f32_t* dx_out, f32_t* d_prenorm_out, f32_t* dNorm_w, f32_t* dNorm_b,
    const T* __restrict__ norm_rstd, const T* __restrict__ centered_out, const T* __restrict__ prenorm_out,
    const T* __restrict__ predrop_out, const T* __restrict__ preact_out, const T* __restrict__ mask,
    uint32_t m, uint32_t n, uint32_t ec,
    uint32_t norm, uint32_t act, f32_t leaky_relu_coeff
);

template<typename T> __device__ void dev_optimiser_update_param(
    T* __restrict__ param, f32_t* master_param, f32_t* dv, f32_t* dm, uint32_t idx,
    f32_t grad_acc, f32_t inv_m, f32_t lr, f32_t max_grad_norm,
    uint32_t optimiser, f32_t beta1, f32_t beta2, f32_t epsilon,
    uint32_t step, uint32_t nesterov, uint32_t regularisation, f32_t regu_coeff, bool apply_regul
);

/**
 * Optimisers: 1 => SGD with momentum, 2 => Adam
 *
 * Regularisation: 1 => L1 Regularisation, 2 => L2 Regularisation
 *
 * @tparam T Either f32_t or f16_t.
 * @param w The weights matrix of the current layer.
 * @param b The biases matrix of the current layer.
 * @param master_w The master weights matrix of the current layer.
 * @param master_b The master biases matrix of the current layer.
 * @param x The input matrix.
 * @param d_prenorm_out The accumulated delta prenorm out matrix of this layer.
 * @param dv_w The dv_w matrix.
 * @param dv_b The dv_b matrix.
 * @param dm_w The dm_w matrix.
 * @param dm_b The dm_b matrix.
 * @param norm_w The normalisation weights matrix of the current layer.
 * @param norm_b The normalisation biases matrix of the current layer.
 * @param master_norm_w The master normalisation weights matrix of the current layer.
 * @param master_norm_b The master normalisation biases matrix of the current layer.
 * @param dNorm_w The accumulated delta norm weights matrix of this layer.
 * @param dNorm_b The accumulated delta norm biases matrix of this layer.
 * @param dv_norm_w The dv_norm_w matrix.
 * @param dv_norm_b The dv_norm_b matrix.
 * @param dm_norm_w The dm_norm_w matrix.
 * @param dm_norm_b The dm_norm_b matrix.
 * @param use_bias If true, biases will be updated. Otherwise, biases will not be updated.
 * @param norm The normalisation mode used.
 * @param m The rows of the error deltas matrix (batch size).
 * @param n The columns of the error deltas matrix (output features).
 * @param wr The rows of the weights matrix (input features).
 * @param lr The learning rate.
 * @param max_grad_norm The absolute maximum gradient value for gradient clipping.
 * @param optimiser The optimiser mode for the linear phase.
 * @param norm_optimiser The optimiser mode for the normalisation phase.
 * @param linear_beta1 The beta1 coefficient for the linear optimiser.
 * @param linear_beta2 The beta2 coefficient for the linear optimiser.
 * @param linear_epsilon A constant for the linear Adam optimiser to prevent zero-division.
 * @param linear_nesterov If set to true, SGD with momentum (for linear weights and biases) will use the nesterov equation.
 * @param norm_beta1 The beta1 coefficient for the norm optimiser.
 * @param norm_beta2 The beta2 coefficient for the norm optimiser.
 * @param norm_epsilon A constant for the norm Adam optimiser to prevent zero-division.
 * @param norm_nesterov If set to true, SGD with momentum (for norm weights and biases) will use the nesterov equation.
 * @param regularisation The regularisation mode used.
 * @param regu_coeff The coefficient for regularisation.
 * @param step The number of step of this iteration.
 */
template<typename T> __device__ void backward_pass_kernel(
    const T* __restrict__ x, T* __restrict__ w, T* __restrict__ b, f32_t* master_w, f32_t* master_b, f32_t* d_prenorm_out,
    f32_t* dv_w, f32_t* dv_b, f32_t* dm_w, f32_t* dm_b,
    T* __restrict__ norm_w, T* __restrict__ norm_b, f32_t* master_norm_w, f32_t* master_norm_b, f32_t* dNorm_w, f32_t* dNorm_b,
    f32_t* dv_norm_w, f32_t* dv_norm_b, f32_t* dm_norm_w, f32_t* dm_norm_b,
    uint32_t use_bias, uint32_t norm,
    uint32_t m, uint32_t n, uint32_t wr, f32_t lr, f32_t max_grad_norm,
    uint32_t optimiser, uint32_t norm_optimiser,
    f32_t linear_beta1, f32_t linear_beta2, f32_t linear_epsilon, uint32_t linear_nesterov,
    f32_t norm_beta1, f32_t norm_beta2, f32_t norm_epsilon, uint32_t norm_nesterov,
    uint32_t regularisation, f32_t regu_coeff, uint32_t step
);

#if __CUDA_ARCH__ >= 700
__device__ void compute_hidden_layer_error_wmma_kernel(
    const f32_t* next_d_prenorm_out, const f32_t* master_w_next, const f32_t* master_norm_w,
    f32_t* dx_out, f32_t* d_prenorm_out, f32_t* dNorm_w, f32_t* dNorm_b,
    const f16_t* norm_rstd, const f16_t* centered_out, const f16_t* prenorm_out,
    const f16_t* predrop_out, const f16_t* preact_out, const f16_t* mask,
    uint32_t m, uint32_t n, uint32_t ec, uint32_t norm, uint32_t act, f32_t leaky_relu_coeff
);
#endif


#endif