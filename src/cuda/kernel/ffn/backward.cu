#pragma once

#ifndef FFN_BACKWARD_CU
#define FFN_BACKWARD_CU

#include "../util.cu"
#include "../math.cu"

constexpr f32_t LOSS_SCALE = 512.0f;
constexpr f32_t LOSS_SCALE_INV = 1.0f / LOSS_SCALE;

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
// threadIdx.x corresponds to n (current weight columns/output features), threadIdx.y corresponds to m (batch size)
template<typename T> __device__ inline void compute_output_layer_error_kernel(
    const T* __restrict__ out, const T* __restrict__ preact_out, const T* __restrict__ target, const f32_t* master_norm_w,
    f32_t* dx_out, f32_t* d_prenorm_out, f32_t* dNorm_w, f32_t* dNorm_b,
    const T* __restrict__ norm_rstd, const T* __restrict__ centered_out, const T* __restrict__ prenorm_out,
    const uint32_t m, const uint32_t n, const uint32_t err_mode,
    const uint32_t norm, const uint32_t act, const f32_t leaky_relu_coeff
) {
    const f32_t inv_m = 1.0f / static_cast<f32_t>(m);
    const f32_t inv_n = 1.0f / static_cast<f32_t>(n);

    const uint32_t b = blockIdx.y * blockDim.y + threadIdx.y;
    if (const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; b < m && col < n) {
        const uint32_t idx = b * n + col;

        f32_t out_v_f32 = static_cast<f32_t>(out[idx]);
        out_v_f32 = dev_activation(act, out_v_f32, leaky_relu_coeff);

        const f32_t target_f32 = static_cast<f32_t>(target[idx]);

        f32_t err_delta_f32 = 0.0f;

        // dL / dZ
        if (err_mode == 0) {
            err_delta_f32 = out_v_f32 - target_f32;

            // a'(Z)
            err_delta_f32 = dev_activation_derivative(
                act, err_delta_f32, out_v_f32, static_cast<f32_t>(preact_out[idx]), leaky_relu_coeff
            );
        } else if ((err_mode == 1 && act == 5) || (err_mode == 2 && act == 1)) {
            err_delta_f32 = out_v_f32 - target_f32;
        }

        dx_out[idx] = err_delta_f32;

        const uint32_t rstd_idx = norm == 3 /* BatchNorm */ ? col : b;

        dev_write_norm_gradients<T>(
            d_prenorm_out, dNorm_w, dNorm_b,
            centered_out, prenorm_out, static_cast<f32_t>(norm_rstd[rstd_idx]),
            err_delta_f32, master_norm_w[col], LOSS_SCALE,
            norm, idx, inv_n, inv_m
        );
    }
}

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
// threadIdx.x corresponds to n (current weight columns/output features), threadIdx.y corresponds to m (batch size)
template<typename T> __device__ inline void compute_hidden_layer_error_kernel(
    const f32_t* next_d_prenorm_out, const f32_t* master_w_next, const f32_t* master_norm_w,
    f32_t* dx_out, f32_t* d_prenorm_out, f32_t* dNorm_w, f32_t* dNorm_b,
    const T* __restrict__ norm_rstd, const T* __restrict__ centered_out, const T* __restrict__ prenorm_out,
    const T* __restrict__ predrop_out, const T* __restrict__ preact_out, const T* __restrict__ mask,
    const uint32_t m, const uint32_t n, const uint32_t ec,
    const uint32_t norm, const uint32_t act, const f32_t leaky_relu_coeff
) {
    const f32_t inv_m = 1.0f / static_cast<f32_t>(m);
    const f32_t inv_n = 1.0f / static_cast<f32_t>(n);

    const uint32_t batch = blockIdx.y * blockDim.y + threadIdx.y;
    if (const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x; batch < m && col < n) {
        f32_t next_layer_sum = 0.0f;

        // Coalesced memory access pattern
        for (uint32_t k = 0; k < ec; k++) {
            next_layer_sum += next_d_prenorm_out[batch * ec + k] * master_w_next[col * ec + k];
        }

        const uint32_t idx = batch * n + col;

        const f32_t derivative = dev_activation_derivative(
            act,
            next_layer_sum,
            static_cast<f32_t>(predrop_out[idx]),
            static_cast<f32_t>(preact_out[idx]),
            leaky_relu_coeff
        );

        // Calculate accumulated gradients
        const f32_t final_delta_f32 = static_cast<f32_t>(mask[idx]) * derivative;
        dx_out[idx] = final_delta_f32;

        const uint32_t rstd_idx = norm == 3 /* BatchNorm */ ? col : batch;

        dev_write_norm_gradients<T>(
            d_prenorm_out, dNorm_w, dNorm_b,
            centered_out, prenorm_out, static_cast<f32_t>(norm_rstd[rstd_idx]),
            final_delta_f32, master_norm_w[col], 1.0f,
            norm, idx, inv_n, inv_m
        );
    }
}

// m = batch size, n = current layer weight columns (also output features), wr = current layer weight rows
template<typename T> __device__ inline void dev_optimiser_update_param(
    T* __restrict__ param, f32_t* master_param, f32_t* dv, f32_t* dm, uint32_t idx,
    const f32_t grad_acc, const f32_t inv_m, const f32_t lr, const f32_t max_grad_norm,
    const uint32_t optimiser, const f32_t beta1, const f32_t beta2, const f32_t epsilon,
    const uint32_t step, const uint32_t nesterov,
    const uint32_t regularisation, const f32_t regu_coeff, const bool apply_regul
) {
    f32_t p_val  = master_param[idx];
    f32_t dv_val = dv[idx];
    f32_t dm_val = dm[idx];

    const f32_t val = grad_acc * inv_m;

    // Regularisation
    if (apply_regul) {
        const f32_t sign = dev_signum<f32_t>(master_param[idx]);
        if (regularisation == 1) p_val -= regu_coeff * sign;           // L1
        if (regularisation == 2) p_val -= 2.0f * regu_coeff * p_val;   // L2
    }

    if (optimiser == 0) {
        // SGD with optional Nesterov momentum
        const f32_t dv_local = beta1 * dv_val + val;
        dv_val = dv_local;

        f32_t update = lr * (nesterov ? val + beta1 * dv_local : dv_local);
        update = CudaMath<f32_t>::clamp(update, -max_grad_norm, max_grad_norm);
        p_val -= update;

    } else if (optimiser == 1) {
        // Adam
        const f32_t dm_local = beta1 * dm_val + (1.0f - beta1) * val;
        const f32_t dv_local = beta2 * dv_val + (1.0f - beta2) * val * val;
        dm_val = dm_local;
        dv_val = dv_local;

        const f32_t m_corrected = dm_local / (1.0f - CudaMath<f32_t>::pow(beta1, static_cast<f32_t>(step)));
        const f32_t v_corrected = dv_local / (1.0f - CudaMath<f32_t>::pow(beta2, static_cast<f32_t>(step)));
        const f32_t denom = CudaMath<f32_t>::sqrt(v_corrected) + epsilon;

        f32_t update = lr * (m_corrected / denom);
        update = CudaMath<f32_t>::clamp(update, -max_grad_norm, max_grad_norm);
        p_val -= update;
    }

    master_param[idx] = p_val;
    param[idx]        = static_cast<T>(p_val);
    dv[idx]           = dv_val;
    dm[idx]           = dm_val;
}

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
// threadIdx.x corresponds to n (current layer weight columns/output features), threadIdx.y corresponds to wr (current layer weight rows/input features)
template<typename T> __device__ inline void backward_pass_kernel(
    const T* __restrict__ x, T* __restrict__ w, T* __restrict__ b, f32_t* master_w, f32_t* master_b, f32_t* d_prenorm_out,
    f32_t* dv_w, f32_t* dv_b, f32_t* dm_w, f32_t* dm_b,
    T* __restrict__ norm_w, T* __restrict__ norm_b, f32_t* master_norm_w, f32_t* master_norm_b, f32_t* dNorm_w, f32_t* dNorm_b,
    f32_t* dv_norm_w, f32_t* dv_norm_b, f32_t* dm_norm_w, f32_t* dm_norm_b,
    const uint32_t use_bias, const uint32_t norm,
    const uint32_t m, const uint32_t n, const uint32_t wr, const f32_t lr, const f32_t max_grad_norm,
    const uint32_t optimiser, const uint32_t norm_optimiser,
    const f32_t linear_beta1, const f32_t linear_beta2, const f32_t linear_epsilon, const uint32_t linear_nesterov,
    const f32_t norm_beta1, const f32_t norm_beta2, const f32_t norm_epsilon, const uint32_t norm_nesterov,
    const uint32_t regularisation, const f32_t regu_coeff, const uint32_t step
) {
    const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= wr || col >= n) return;

    const f32_t inv_m = 1.0f / static_cast<f32_t>(m);

    f32_t dW_acc      = 0.0f;
    f32_t dB_acc      = 0.0f;
    f32_t dNorm_w_acc = 0.0f;
    f32_t dNorm_b_acc = 0.0f;

    // Finish accumulating weights
    for (uint32_t bi = 0; bi < m; bi++) {
        const uint32_t b_idx        = bi * n + col;
        const f32_t    d_prenorm    = d_prenorm_out[b_idx];

        dW_acc += static_cast<f32_t>(x[bi * wr + row]) * d_prenorm;

        if (row == 0) {
            dB_acc += d_prenorm;
            if (norm > 0) {
                dNorm_w_acc += dNorm_w[b_idx];
                if (norm > 1)
                    dNorm_b_acc += dNorm_b[b_idx];
            }
        }
    }

    // Update parameters using optimisers
    if (row == 0) {
        if (use_bias) {
            dev_optimiser_update_param<T>(
                b, master_b, dv_b, dm_b, col,
                dB_acc * LOSS_SCALE_INV, inv_m, lr, max_grad_norm, optimiser,
                linear_beta1, linear_beta2, linear_epsilon, step, linear_nesterov,
                regularisation, regu_coeff, false
            );
        }

        if (norm > 0) {
            dev_optimiser_update_param<T>(
                norm_w, master_norm_w, dv_norm_w, dm_norm_w, col,
                dNorm_w_acc * LOSS_SCALE_INV, inv_m, lr, max_grad_norm, norm_optimiser,
                norm_beta1, norm_beta2, norm_epsilon, step, norm_nesterov,
                regularisation, regu_coeff, false
            );

            if (norm > 1) {
                dev_optimiser_update_param<T>(
                    norm_b, master_norm_b, dv_norm_b, dm_norm_b, col,
                    dNorm_b_acc * LOSS_SCALE_INV, inv_m, lr, max_grad_norm, norm_optimiser,
                    norm_beta1, norm_beta2, norm_epsilon, step, norm_nesterov,
                    regularisation, regu_coeff, false
                );
            }
        }
    }

    const uint32_t idx = row * n + col;
    dev_optimiser_update_param<T>(
        w, master_w, dv_w, dm_w, idx,
        dW_acc * LOSS_SCALE_INV, inv_m, lr, max_grad_norm, optimiser,
        linear_beta1, linear_beta2, linear_epsilon, step, linear_nesterov,
        regularisation, regu_coeff, true
    );
}

#endif