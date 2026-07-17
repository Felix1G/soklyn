#pragma once

#ifndef FFN_BACKWARD_CU
#define FFN_BACKWARD_CU

#include "backward.cuh"

constexpr f32_t LOSS_SCALE = 512.0f;
constexpr f32_t LOSS_SCALE_INV = 1.0f / LOSS_SCALE;

// threadIdx.x corresponds to n (current weight columns/output features), threadIdx.y corresponds to m (batch size)
template<typename T> __device__ void compute_output_layer_error_kernel(
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

// threadIdx.x corresponds to n (current weight columns/output features), threadIdx.y corresponds to m (batch size)
template<typename T> __device__ void compute_hidden_layer_error_kernel(
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
template<typename T> __device__ void dev_optimiser_update_param(
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

// threadIdx.x corresponds to n (current layer weight columns/output features), threadIdx.y corresponds to wr (current layer weight rows/input features)
template<typename T> __device__ void backward_pass_kernel(
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