extern "C" __global__ void memset_kernel(float* dst, float v, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        dst[idx] = v;
    }
}

__device__ __inline__ float dev_gen_random_float(unsigned int col, unsigned int row, unsigned int seed) {
    unsigned int state = (row * 1103515245) ^ (col * 6364136223846793005U) ^ seed;
    state = state ^ (state >> 17);
    state = state * 1103515245 + 12345;
    state = state ^ (state >> 15);
    return ((float)(state) / (float)(0xFFFFFFFFU));
}

__device__ __inline__ float dev_signum(float x) {
    return (x > 0.0f) - (x < 0.0f);
}

extern "C" __device__ float dev_sgemm(float* a, float* b, int m, int n, int p, int tile_dim, int row, int col) {
    extern __shared__ float shared_mem[];

    float* tile_A = shared_mem;
    float* tile_B = &shared_mem[tile_dim * tile_dim];

    float sum = 0.0f;

    for (int t = 0; t < (n + tile_dim - 1) / tile_dim; ++t) {
        int tile_offset = t * tile_dim;

        if (row < m && (tile_offset + threadIdx.x) < n) {
            tile_A[threadIdx.y * tile_dim + threadIdx.x] = a[row * n + tile_offset + threadIdx.x];
        } else {
            tile_A[threadIdx.y * tile_dim + threadIdx.x] = 0.0f;
        }

        if (col < p && (tile_offset + threadIdx.y) < n) {
            tile_B[threadIdx.y * tile_dim + threadIdx.x] = b[(tile_offset + threadIdx.y) * p + col];
        } else {
            tile_B[threadIdx.y * tile_dim + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Dot product
        for (int k = 0; k < tile_dim; ++k) {
            sum += tile_A[threadIdx.y * tile_dim + k] * tile_B[k * tile_dim + threadIdx.x];
        }

        __syncthreads();
    }

    return sum;
}

extern "C" __global__ void sgemm_kernel(float* a, float* b, float* c, int m, int n, int p, int tile_dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = dev_sgemm(a, b, m, n, p, tile_dim, row, col);

    if (row < m && col < p) {
        c[row * p + col] = sum;
    }
}

extern "C" __global__  void sgeam_kernel(float* a, float* b, float* c, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        int idx = row * n + col;
        c[idx] = a[idx] + b[idx];
    }
}

// 0: None, 1: Sigmoid, 2: ReLU, 3: LeakyReLU, 4: Tanh, 5: Softmax, 6: SiLU, 7: Mish
extern "C" __device__ float dev_activation(int act, float sum, float leaky_relu_coeff) {
    float new_sum = sum;
    if (act == 1) {
        new_sum = 1.0f / (1.0f + expf(-sum));
    } else if (act == 2) {
        new_sum = fmaxf(0.0f, sum);
    } else if (act == 3) {
        new_sum = (sum > 0.0f ? sum : (leaky_relu_coeff * sum));
    } else if (act == 4) {
        new_sum = tanhf(sum);
    } else if (act == 6) {
        new_sum = sum / (1.0f + expf(-sum));
    } else if (act == 7) {
        float e_x = expf(sum);

        // softplus
        float sp = (sum > 20.0f) ? sum : logf(1.0f + e_x);

        float e_sp = expf(sp);
        float e_sp_inv = 1.0f / e_sp;
        float e_tanh = (e_sp - e_sp_inv) / (e_sp + e_sp_inv);

        new_sum = sum * e_tanh;
    }
    return new_sum;
}

extern "C" __device__ float dev_activation_derivative(int act, float err_delta, float out_v, float in_x, float leaky_relu_coeff) {
    float err = err_delta;
    if (act == 1) {
        err *= out_v * (1.0 - out_v);
    } else if (act == 2) {
        err *= out_v > 0.0f;
    } else if (act == 3) {
        err *= (out_v > 0.0f) ? 1.0f : leaky_relu_coeff;
    } else if (act == 4) {
        err *= (1.0 - out_v * out_v);
    } else if (act == 6) {
        float sig_x = 1.0f / (1.0f + expf(-in_x));
        err *= sig_x + out_v * (1.0f - sig_x);
    } else if (act == 7) {
        float e_x = expf(in_x);
        float sp = (in_x > 20.0f) ? in_x : logf(1.0f + e_x);
        float e_sp = expf(sp);
        float e_sp_inv = 1.0f / e_sp;
        float e_tanh = (e_sp - e_sp_inv) / (e_sp + e_sp_inv);

        float sig_x = e_x / (1.0f + e_x);

        float e_sech2 = 1.0f - (e_tanh * e_tanh);
        err *= e_tanh + in_x * sig_x * e_sech2;
    }
    return err;
}

// in ---linear(w,b)---> prenorm (or preact if there is no normalisation, set in the host)
// output size => (m, wc)
// m = batch size, n = input features to this linear
// wr = previous layer neurons (which is n), wc = current layer neurons
// bias column = wc
// act = activation
// threadIdx.x: wc, threadIdx.y: m
extern "C" __global__ void forward_pass_0_kernel(
    float* prenorm_out, float* in, float* w, float* b, int use_bias, int m, int n, int wc, int tile_dim
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // weight multiplication
    float sum = dev_sgemm(in, w, m, n, wc, tile_dim, row, col);

    if (row < m && col < wc) {
        if (use_bias) sum += b[col]; // bias addition

        int idx = row * wc + col;
        prenorm_out[idx] = sum;
    }
}

extern "C" __device__ void dev_block_stride_sum(int tid, float* shared_sum) {
    __syncthreads(); // ensure shared_sum has been fully written
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared_sum[tid] += shared_sum[tid + stride];
        __syncthreads();
    }
}

// prenorm ---centering---> centered ---normalisation---> preact
// Each block owns one entire row (norm != BatchNorm) or column (norm == BatchNorm)
extern "C" __global__ void forward_pass_1_kernel(
    float* preact_out, float* centered_out, float* prenorm_out, float* norm_w, float* norm_b, float* norm_rstd, int m, int wc, int norm
) {
    extern __shared__ float shared_sum[];
    int bid = blockIdx.x;
    if ((norm == 3 && bid >= wc) || (norm != 3 && bid >= m)) return;

    int tid = threadIdx.x;

    if (norm == 1) { // RMSNorm
        int row         = bid;
        float sq_sum    = 0.0f;

        // include values where indices exceed this block size
        for (int col = tid; col < wc; col += blockDim.x) {
            float val   = prenorm_out[row * wc + col];
            sq_sum      += val * val; // Squaring following RMS formula
        }

        // Reduce sums to index 0
        shared_sum[tid] = sq_sum;
        dev_block_stride_sum(tid, shared_sum);

        // Finish calculation
        float rstd = rsqrtf((shared_sum[0] / (float)wc) + 1e-6f);

        for (int col = tid; col < wc; col += blockDim.x) {
            int idx             = row * wc + col;
            norm_rstd[idx]      = rstd;
            centered_out[idx]   = prenorm_out[idx];
            preact_out[idx]     = prenorm_out[idx] * rstd * norm_w[col];
        }
    } else if (norm == 2 || norm == 3) { // LayerNorm or BatchNorm
        int n       = (norm == 2) ? wc : m;
        int stride  = blockDim.x;

        // ------------- FIND MEAN -------------
        float total_sum = 0.0f;

        // include values where indices exceed this block size
        for (int i = tid; i < n; i += stride) {
            int idx     = (norm == 2) ? (bid * wc + i) : (i * wc + bid);
            total_sum   += prenorm_out[idx];
        }

        shared_sum[tid] = total_sum;
        dev_block_stride_sum(tid, shared_sum);
        float mean = shared_sum[0] / (float)n;

        // ------------- FIND VARIANCE -------------
        float std_sum = 0.0f;

        // include values where indices exceed this block size
        for (int i = tid; i < n; i += stride) {
            int idx     = (norm == 2) ? (bid * wc + i) : (i * wc + bid);
            float val   = prenorm_out[idx] - mean;
            std_sum     += val * val;
        }

        shared_sum[tid] = std_sum;
        dev_block_stride_sum(tid, shared_sum);
        float rstd = rsqrtf(shared_sum[0] / (float)n + 1e-6f);

        // ------------- FINISH CALCULATIONS -------------
        for (int i = tid; i < n; i += stride) {
            int idx             = (norm == 2) ? (bid * wc + i) : (i * wc + bid);
            int col             = (norm == 2) ? i : bid;
            float val           = prenorm_out[idx] - mean;
            norm_rstd[idx]      = rstd;
            centered_out[idx]   = val;
            preact_out[idx]     = val * rstd * norm_w[col] + norm_b[col];
        }
    }
}

// preact ---activation---> predrop ---dropout/masking---> out
// threadIdx.x: wc, threadIdx.y: m
extern "C" __global__ void forward_pass_2_kernel(
    float* out, float* predrop_out, float* preact_out, float* mask,
    int m, int wc, int act,
    float leaky_relu_coeff, int is_training, float mask_coeff, unsigned int seed
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < wc) {
        int idx = row * wc + col;
        float sum = preact_out[idx];

        // activation function
        sum = dev_activation(act, sum, leaky_relu_coeff);

        // dropout
        float mask_v = 1.0f;
        if (is_training) {
            if (dev_gen_random_float(col, row, seed + row * 7) >= mask_coeff) {
                mask_v = 1.0f / (1.0f - mask_coeff);
            } else {
                mask_v = 0.0f;
            }

            mask[idx] = mask_v;
        }

        predrop_out[idx] = sum;
        out[idx] = mask_v * sum;
    }
}

// each block corresponds to a row, each thread simultaneously handles each column
// m: out rows, n: out columns
// threadIdx.x: m
extern "C" __global__ void softmax_kernel(float* out, int m, int n) {
    int row = blockIdx.x;
    if (row >= m) return;

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    // ========================================================================
    // MAX VALUES (prevents overflow)
    // ========================================================================
    float row_max = -INFINITY;
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        row_max = fmaxf(row_max, out[row * n + col]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
    }

    if (lane == 0) shared[wid] = row_max; // warp maximum
    __syncthreads();

    row_max = (threadIdx.x < num_warps) ? shared[lane] : -INFINITY;
    if (wid == 0) { // block maximum
        for (int offset = 16; offset > 0; offset >>= 1) {
            row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
        }
    }

    __shared__ float block_global_max;
    if (threadIdx.x == 0) {
        block_global_max = row_max; // broadcasting
    }
    __syncthreads();

    // ========================================================================
    // SUM VALUES
    // ========================================================================
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        local_sum += expf(out[row * n + col] - block_global_max);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) shared[wid] = local_sum; // warp sum
    __syncthreads();

    local_sum = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) { // block sum
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
    }

    __shared__ float block_global_sum;
    if (threadIdx.x == 0) {
        block_global_sum = local_sum; // broadcasting
    }
    __syncthreads();

    // ========================================================================
    // CALCULATE SOFTMAX
    // ========================================================================
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
        int idx = row * n + col;
        float prev_sum = out[idx]; 
        float sum = expf(prev_sum - block_global_max) / block_global_sum;

        out[idx] = sum;
    }
}

// m = batch size, n = current weight columns (also output features)
// threadIdx.x: n, threadIdx.y: m
extern "C" __global__ void compute_output_layer_error_kernel(
    float* out, float* preact_out, float* dx_out, float* target, int m, int n, int err_mode, int act, float leaky_relu_coeff
) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float err_delta = 0.0f;
    if (b < m && col < n) {
        int idx = b * n + col;
        float out_v = out[idx];
        out_v = dev_activation(act, out_v, leaky_relu_coeff);

        // dL / dZ
        if (err_mode == 0) {
            err_delta = out_v - target[idx];

            // a'(Z)
            err_delta = dev_activation_derivative(act, err_delta, out_v, preact_out[idx], leaky_relu_coeff);
        } else if ((err_mode == 1 && act == 5) || (err_mode == 2 && act == 1)) {
            err_delta = out_v - target[idx];
        }

        dx_out[idx] = err_delta;
    }
}

// m = batch size, n = current layer weight columns, ec = next error columns
// thread x: n, thread y: m
extern "C" __global__ void compute_hidden_layer_error_kernel(
    float* dx_out, float* err_next, float* w_next, float* out, 
    float* predrop_out, float* preact_out, float* mask,
    int m, int n, int ec, int act, float leaky_relu_coeff
) {
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < m && col < n) {
        float next_layer_sum = 0.0f;

        // Coalesced memory access pattern
        for (int k = 0; k < ec; k++) {
            next_layer_sum += err_next[b * ec + k] * w_next[col * ec + k];
        }

        int idx = b * n + col;
        float out_v = predrop_out[idx];
        dx_out[idx] = mask[idx] * dev_activation_derivative(act, next_layer_sum, out_v, preact_out[idx], leaky_relu_coeff);
    }
}

__device__ inline void dev_optimiser_update_param(
    float* param, float* dv, float* dm, int idx,
    float grad_acc, float inv_m, float lr, float max_grad_norm, int optimiser,
    float beta1, float beta2, float epsilon, int step, int nesterov, 
    int regularisation, float regu_coeff, bool apply_regul
) {
    float val = grad_acc * inv_m;

    val = fminf(max_grad_norm, fmaxf(val, -max_grad_norm));

    float is_l1 = (apply_regul && regularisation == 1);
    float is_l2 = (apply_regul && regularisation == 2);
    param[idx] -= is_l1 * regu_coeff * dev_signum(param[idx]);
    param[idx] -= is_l2 * 2.0f * regu_coeff * param[idx];
    
    // SGD
    if (optimiser == 0) {
        float dv_local = beta1 * dv[idx] + val;
        dv[idx] = dv_local;
        param[idx] -= lr * (nesterov ? (val + beta1 * dv_local) : dv_local);
    // Adam
    } else if (optimiser == 1) {
        float dm_local = beta1 * dm[idx] + (1.0f - beta1) * val;
        float dv_local = beta2 * dv[idx] + (1.0f - beta2) * val * val;

        dm[idx] = dm_local;
        dv[idx] = dv_local;

        float m_corrected = dm_local / (1.0f - powf(beta1, step));
        float v_corrected = dv_local / (1.0f - powf(beta2, step));

        if (v_corrected < 1e-15f) {
            v_corrected = 0.0f;
        }

        float denom = sqrtf(v_corrected) + epsilon;
        if (denom < 1e-8f) denom = 1e-8f;

        param[idx] -= lr * (m_corrected / denom);
    }
}

// TODO backward pass for BatchNorm has not been tested on inference.
// m = batch size, n = current layer weight columns (also output features), wr = current layer weight rows
// thread x: n, thread y: wr
extern "C" __global__ void backward_pass_kernel(
    float* err, float* w, float* b, float* x, float* preact_out,
    float* dv_w, float* dv_b, float* dm_w, float* dm_b, 
    float* norm_w, float* norm_b, float* norm_rstd, float* centered_out, float* prenorm_out,
    float* dv_norm_w, float* dv_norm_b, float* dm_norm_w, float* dm_norm_b, 
    int use_bias, int norm, int m, int n, int wr, float lr, float max_grad_norm, 
    int optimiser, int norm_optimiser, float beta1, int nesterov, 
    float beta2, float epsilon, int regularisation, float regu_coeff, int step
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= wr || col >= n) return;

    float dNorm_w_acc = 0.0f;
    float dNorm_b_acc = 0.0f;
    float dB_acc = 0.0f;
    float dW_acc = 0.0f;

    float inv_m = 1.0f / (float)m;
    float inv_n = 1.0f / (float)n;
    float inv_size = (norm == 2) ? inv_n : inv_m;
    float size_const = 1.0f - inv_size;

    for (int bi = 0; bi < m; bi++) {
        int b_idx = bi * n + col;
        float err_delta = err[b_idx];
        float d_prenorm = err_delta; // passthrough for norm == 0

        if (norm == 1) { // RMSNorm
            float rstd = norm_rstd[b_idx];
            float pval = prenorm_out[b_idx];
            if (row == 0)
                dNorm_w_acc += err_delta * pval * rstd;
            d_prenorm = err_delta * norm_w[col] *
                        (rstd - (rstd * rstd * rstd * pval * pval) * inv_n);
        } else if (norm == 2 || norm == 3) { // LayerNorm or BatchNorm
            float rstd = norm_rstd[b_idx];
            float cval = centered_out[b_idx];
            if (row == 0) {
                dNorm_b_acc += err_delta;
                dNorm_w_acc += err_delta * cval * rstd;
            }
            d_prenorm = err_delta * norm_w[col] * size_const *
                        (rstd - (rstd * rstd * rstd * cval * cval) * inv_size);
        }

        dW_acc += x[bi * wr + row] * d_prenorm;
        if (row == 0)
            dB_acc += d_prenorm;
    }

    if (row == 0) {
        if (use_bias) dev_optimiser_update_param(b, dv_b, dm_b, col, dB_acc, inv_m, lr, max_grad_norm, optimiser, beta1, beta2, epsilon, step, nesterov, regularisation, regu_coeff, false);

        if (norm > 0) {
            dev_optimiser_update_param(norm_w, dv_norm_w, dm_norm_w, col, dNorm_w_acc, inv_m, lr, max_grad_norm, norm_optimiser, beta1, beta2, epsilon, step, nesterov, regularisation, regu_coeff, false);

            if (norm > 1) {
                dev_optimiser_update_param(norm_b, dv_norm_b, dm_norm_b, col, dNorm_b_acc, inv_m, lr, max_grad_norm, norm_optimiser, beta1, beta2, epsilon, step, nesterov, regularisation, regu_coeff, false);
            }
        }
    }

    int idx = row * n + col;
    dev_optimiser_update_param(w, dv_w, dm_w, idx, dW_acc, inv_m, lr, max_grad_norm, optimiser, beta1, beta2, epsilon, step, nesterov, regularisation, regu_coeff, true);
}