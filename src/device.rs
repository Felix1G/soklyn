use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::driver::sys::CUctx_flags;
use cudarc::nvrtc::Ptx;
use crate::nn::functions::{Activation, ErrorFunc, Normalisation, Optimiser, Regularisation};
use crate::nn::network::DenseBlock;
use crate::util::{scramble_seed, Tensor};

/// Carries the context of the connection to a GPU device.
#[allow(dead_code)]
pub struct GpuContext {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    memset_func: CudaFunction,
    sgemm_func: CudaFunction, // A * B
    sgeam_func: CudaFunction, // A + B
    forward_pass_func0: CudaFunction, // Z = WX + B
    forward_pass_func1: CudaFunction, // Normalisation
    forward_pass_func2: CudaFunction, // Activation and dropout
    softmax_func: CudaFunction,
    compute_output_layer_error_func: CudaFunction,
    compute_hidden_layer_error_func: CudaFunction,
    backward_pass_func: CudaFunction,
    tile_dim: u32,
    tile_dim_minus_1: u32,
    tile_dim_2: u32,
    tile_dim_2_minus_1: u32,
}

#[allow(dead_code)]
impl GpuContext {
    fn new_from_context(context: Arc<CudaContext>, tile_dim: u32) -> Self {
        let nn_math_path = include_str!(concat!(env!("OUT_DIR"), "/cuda/nn_math.ptx"));
        let nn_math_src = Ptx::from_src(nn_math_path);
        let nn_math_mod = context.load_module(nn_math_src).expect("Failed to load module 'nn_math.ptx'");

        let load_kernel = |name: &str| {
            nn_math_mod
                .load_function(name)
                .unwrap_or_else(|e| panic!("Failed to load CUDA function '{}': {:?}", name, e))
        };

        let memset_func = load_kernel("memset_kernel");
        let sgemm_func = load_kernel("sgemm_kernel");
        let sgeam_func = load_kernel("sgeam_kernel");
        let forward_pass_func0 = load_kernel("forward_pass_0_kernel");
        let forward_pass_func1 = load_kernel("forward_pass_1_kernel");
        let forward_pass_func2 = load_kernel("forward_pass_2_kernel");
        let softmax_func = load_kernel("softmax_kernel");
        let compute_output_layer_error_func = load_kernel("compute_output_layer_error_kernel");
        let compute_hidden_layer_error_func = load_kernel("compute_hidden_layer_error_kernel");
        let backward_pass_func = load_kernel("backward_pass_kernel");

        let stream = CudaContext::new_stream(&context).expect("Failed to create stream");
        context.set_flags(CUctx_flags::CU_CTX_SCHED_SPIN).expect("Failed to set context flags");

        let tile_dim_2 = tile_dim * tile_dim;

        Self {
            context,
            stream,
            memset_func,
            sgemm_func,
            sgeam_func,
            forward_pass_func0,
            forward_pass_func1,
            forward_pass_func2,
            softmax_func,
            compute_output_layer_error_func,
            compute_hidden_layer_error_func,
            backward_pass_func,
            tile_dim,
            tile_dim_minus_1: tile_dim - 1,
            tile_dim_2,
            tile_dim_2_minus_1: tile_dim_2 - 1,
        }
    }

    /// Initialises a connection to the first available NVIDIA graphics card.
    ///
    /// # Arguments
    /// * `tile_dim` - Tile Dimension, the size of each block `(tile_dim, tile_dim)`.
    /// Recommended values are `16` or `32`.
    ///
    /// # Panics
    ///
    /// Panics if no compatible NVIDIA GPU is found, or if the CUDA driver
    /// is missing/incompatible.
    pub fn new(tile_dim: u32) -> Self {
        let context = CudaContext::new(0).expect("Failed to create CUDA context");
        Self::new_from_context(context, tile_dim)
    }

    /// Initialises a connection to an available NVIDIA graphics card.
    ///
    /// # Arguments
    /// * `ordinal` - The NVIDIA device index. If you only have one device, set this to `0`.
    /// * `tile_dim` - Tile Dimension, the size of each block `(tile_dim, tile_dim)`.
    /// Recommended values are `16` or `32`.
    ///
    /// # Panics
    ///
    /// Panics if no compatible NVIDIA GPU is found, or if the CUDA driver
    /// is missing/incompatible.
    pub fn with_device_index(ordinal: usize, tile_dim: u32) -> Self {
        let context = CudaContext::new(ordinal).expect("Failed to create CUDA context");
        Self::new_from_context(context, tile_dim)
    }

    pub(crate) fn get_context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub(crate) fn get_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    fn calculate_cfg2d(&self, x: usize, y: usize, mem: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: ((x as u32 + self.tile_dim_minus_1) / self.tile_dim,
                       (y as u32 + self.tile_dim_minus_1) / self.tile_dim, 1),
            block_dim: (self.tile_dim, self.tile_dim, 1),
            shared_mem_bytes: mem,
        }
    }

    pub(crate) fn gpu_memset(&self, dst: &CudaSlice<f32>, v: f32) {
        let len = dst.len();
        let len_i32 = len as i32;

        let mut builder = self.stream.launch_builder(&self.memset_func);
        builder.arg(dst).arg(&v).arg(&len_i32);

        let cfg = LaunchConfig {
            grid_dim: ((len as u32 + self.tile_dim_2_minus_1) / self.tile_dim_2, 1, 1),
            block_dim: (self.tile_dim_2, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            builder.launch(cfg).expect("memset launch failed.");
        }

        self.stream.synchronize().unwrap();
    }

    // Single-precision General Matrix Multiply
    pub(crate) fn gpu_matrix_mul(&self,
                                 a_dev: &CudaSlice<f32>, m: usize, n: usize,
                                 b_dev: &CudaSlice<f32>, p: usize,
                                 c_dev: &mut CudaSlice<f32>) {
        let mut builder = self.stream.launch_builder(&self.sgemm_func);
        builder.arg(a_dev).arg(b_dev).arg(c_dev);
        builder.arg(&m).arg(&n).arg(&p);
        builder.arg(&self.tile_dim);

        let cfg = self.calculate_cfg2d(p, m, 2 * self.tile_dim_2 * 4);

        unsafe {
            builder.launch(cfg).expect("sgemm matrix multiplication launch failed.");
        }

        self.stream.synchronize().unwrap();
    }

    pub(crate) fn gpu_matrix_add(&self,
                                 a_dev: &CudaSlice<f32>, m: usize, n: usize,
                                 b_dev: &CudaSlice<f32>,
                                 c_dev: &mut CudaSlice<f32>) {
        let mut builder = self.stream.launch_builder(&self.sgeam_func);
        builder.arg(a_dev).arg(b_dev).arg(c_dev);
        builder.arg(&m).arg(&n);

        let cfg = self.calculate_cfg2d(n, m, 0);

        unsafe {
            builder.launch(cfg).expect("smat matrix addition launch failed.");
        }

        self.stream.synchronize().unwrap();
    }

    pub(crate) fn gpu_forward_pass(&self, cur_layer: &DenseBlock, input: &Tensor, batch_size: usize,
                                   leaky_relu_coeff: f32, is_training: bool, step: usize) {
        let wc = cur_layer.get_weights().cols();
        let norm = cur_layer.get_normalisation();
        let act = cur_layer.get_activation();
        let mask_coeff = cur_layer.get_mask_coeff();
        let seed = scramble_seed(step as u32, cur_layer.get_weights().get_id() as u32);
        let use_bias = *norm != Normalisation::BatchNorm;

        let use_bias_i32 = use_bias as i32;
        let m_i32 = batch_size as i32;
        let n_i32 = input.cols() as i32;
        let wc_i32 = wc as i32;
        let norm_mode_i32 = norm.ordinal() as i32;
        let act_mode_i32 = act.ordinal() as i32;
        let is_training_i32 = is_training as i32;

        let mut builder = self.stream.launch_builder(&self.forward_pass_func0);
        builder
            .arg(
                if *norm == Normalisation::Disabled { cur_layer.get_preact_outputs().get_data() }
                else { cur_layer.get_prenorm_outputs().get_data() }
            )
            .arg(input.get_data())
            .arg(cur_layer.get_weights().get_data())
            .arg(cur_layer.get_biases().get_data())
            .arg(&use_bias_i32)
            .arg(&m_i32)
            .arg(&n_i32)
            .arg(&wc_i32)
            .arg(&self.tile_dim);

        let cfg = self.calculate_cfg2d(wc, batch_size, 2 * self.tile_dim_2 * 4);

        unsafe {
            builder.launch(cfg).expect("Feed forward network forward pass layer function launch failed.");
        }

        self.stream.synchronize().expect("Failed to synchronize primary forward pass stream.");

        // Normalisation
        if *norm != Normalisation::Disabled {
            builder = self.stream.launch_builder(&self.forward_pass_func1);
            builder
                .arg(cur_layer.get_preact_outputs().get_data())
                .arg(cur_layer.get_centered_outputs().get_data())
                .arg(cur_layer.get_prenorm_outputs().get_data())
                .arg(cur_layer.get_norm_weights().get_data())
                .arg(cur_layer.get_norm_biases().get_data())
                .arg(cur_layer.get_norm_rstd().get_data())
                .arg(&m_i32)
                .arg(&wc_i32)
                .arg(&norm_mode_i32);

            let grid_x = if *norm == Normalisation::BatchNorm { wc } else { batch_size };

            let cfg = LaunchConfig {
                grid_dim: (grid_x as u32, 1, 1),
                block_dim: (self.tile_dim_2, 1, 1),
                shared_mem_bytes: self.tile_dim_2 * 4
            };

            unsafe {
                builder.launch(cfg).expect("Feed forward network forward pass layer function launch failed.");
            }

            self.stream.synchronize().expect("Failed to synchronize primary forward pass stream.");
        }

        builder = self.stream.launch_builder(&self.forward_pass_func2);
        builder
            .arg(cur_layer.get_outputs().get_data())
            .arg(cur_layer.get_predrop_outputs().get_data())
            .arg(cur_layer.get_preact_outputs().get_data())
            .arg(cur_layer.get_masks().get_data())
            .arg(&m_i32)
            .arg(&wc_i32)
            .arg(&act_mode_i32)
            .arg(&leaky_relu_coeff)
            .arg(&is_training_i32)
            .arg(&mask_coeff)
            .arg(&seed);

        let cfg = self.calculate_cfg2d(wc, batch_size, 0);

        unsafe {
            builder.launch(cfg).expect("Feed forward network forward pass layer function launch failed.");
        }

        self.stream.synchronize().expect("Failed to synchronize primary forward pass stream.");
    }

    pub(crate) fn gpu_compute_output_layer_error(&self, cur_layer: &DenseBlock, target: &Tensor,
                                                 err_mode: ErrorFunc, activation: Activation) {
        let out_tensor = cur_layer.get_outputs();
        let m = out_tensor.rows();
        let n = cur_layer.get_weights().cols();

        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let err_mode_i32 = err_mode as i32;
        let act_mode_i32 = activation.ordinal() as i32;

        let leaky_relu_coeff = match activation {
            Activation::LeakyReLU(value) => value,
            _ => 0.0,
        };

        // Run softmax if needed first
        if activation == Activation::Softmax {
            let cols_i32 = out_tensor.cols() as i32;
            let mut softmax_builder = self.stream.launch_builder(&self.softmax_func);
            softmax_builder
                .arg(out_tensor.get_data())
                .arg(&m_i32)
                .arg(&cols_i32);

            let softmax_cfg = LaunchConfig {
                grid_dim: (m as u32, 1, 1),
                block_dim: (self.tile_dim_2, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                softmax_builder.launch(softmax_cfg).expect("Softmax function launch failed.");
            }

            self.stream.synchronize().expect("Failed to synchronize softmax stream.");
        }

        let mut err_builder = self.stream.launch_builder(&self.compute_output_layer_error_func);
        err_builder
            .arg(out_tensor.get_data())
            .arg(cur_layer.get_preact_outputs().get_data())
            .arg(cur_layer.get_grads().get_data())
            .arg(target.get_data())
            .arg(&m_i32)
            .arg(&n_i32)
            .arg(&err_mode_i32)
            .arg(&act_mode_i32)
            .arg(&leaky_relu_coeff);

        let cfg = self.calculate_cfg2d(n, m, 0);

        unsafe {
            err_builder.launch(cfg).expect("Feed forward network compute output layer error function launch failed.");
        }
    }

    pub(crate) fn gpu_backward_pass(&self, cur_layer: &DenseBlock, optimiser: &Optimiser, norm_optimiser: &Optimiser, input: &Tensor,
                                    batch_size: usize, lr: f32, max_grad_norm: f32, step: usize) {
        let n = cur_layer.get_weights().cols();
        let wr = cur_layer.get_weights().rows();
        let norm = cur_layer.get_normalisation();
        let regularisation = cur_layer.get_regularisation();
        let use_bias = *norm != Normalisation::BatchNorm;

        let use_bias_i32 = use_bias as i32;
        let norm_mode_i32 = norm.ordinal() as i32;
        let m_i32 = batch_size as i32;
        let n_i32 = n as i32;
        let wr_i32 = wr as i32;
        let optimiser_i32 = optimiser.ordinal() as i32;
        let norm_optimiser_i32 = norm_optimiser.ordinal() as i32;
        let regu_mode_i32 = regularisation.ordinal() as i32;
        let step_i32 = step as i32;

        let (beta1, beta2, epsilon, nesterov) = match optimiser {
            Optimiser::SGD(b1, nest) => {
                (*b1, 0.0, 0.0, if *nest { 1 } else { 0 })
            }
            Optimiser::Adam(b1, b2, eps) => {
                (*b1, *b2, *eps, 0)
            }
        };

        let regu_coeff = match regularisation {
            Regularisation::None => 0.0,
            Regularisation::L1Regular(coeff) => *coeff,
            Regularisation::L2Regular(coeff) => *coeff,
        };

        let mut builder = self.stream.launch_builder(&self.backward_pass_func);
        builder
            .arg(cur_layer.get_grads().get_data())
            .arg(cur_layer.get_weights().get_data())
            .arg(cur_layer.get_biases().get_data())
            .arg(input.get_data())
            .arg(cur_layer.get_preact_outputs().get_data())
            .arg(cur_layer.get_dv_weights().get_data())
            .arg(cur_layer.get_dv_biases().get_data())
            .arg(cur_layer.get_dm_weights().get_data())
            .arg(cur_layer.get_dm_biases().get_data())
            .arg(cur_layer.get_norm_weights().get_data())
            .arg(cur_layer.get_norm_biases().get_data())
            .arg(cur_layer.get_norm_rstd().get_data())
            .arg(cur_layer.get_centered_outputs().get_data())
            .arg(cur_layer.get_prenorm_outputs().get_data())
            .arg(cur_layer.get_dv_norm_weights().get_data())
            .arg(cur_layer.get_dv_norm_biases().get_data())
            .arg(cur_layer.get_dm_norm_weights().get_data())
            .arg(cur_layer.get_dm_norm_biases().get_data())
            .arg(&use_bias_i32)
            .arg(&norm_mode_i32)
            .arg(&m_i32)
            .arg(&n_i32)
            .arg(&wr_i32)
            .arg(&lr)
            .arg(&max_grad_norm)
            .arg(&optimiser_i32)
            .arg(&norm_optimiser_i32)
            .arg(&beta1)
            .arg(&nesterov)
            .arg(&beta2)
            .arg(&epsilon)
            .arg(&regu_mode_i32)
            .arg(&regu_coeff)
            .arg(&step_i32);

        let cfg = self.calculate_cfg2d(n, wr, 0);

        unsafe {
            builder.launch(cfg).expect("Feed forward network back pass (hidden) layer function launch failed.");
        }

        self.stream.synchronize().expect("Failed to synchronize hidden layer backward pass stream.");
    }

    pub(crate) fn gpu_hidden_layer_backpass(&self, cur_layer: &DenseBlock, optimiser: &Optimiser, norm_optimiser: &Optimiser,
                                            next_e: &Tensor, next_w: &Tensor, input: &Tensor,
                                            batch_size: usize, lr: f32, max_grad_norm: f32,
                                            activation: &Activation, step: usize) {
        let m = next_e.rows();
        let n = cur_layer.get_weights().cols();
        let ec = next_e.cols();

        let m_i32 = m as i32;
        let n_i32 = n as i32;
        let ec_i32 = ec as i32;
        let act_mode_i32 = activation.ordinal() as i32;

        let leaky_relu_coeff = match activation {
            Activation::LeakyReLU(value) => *value,
            _ => 0.0,
        };

        let mut err_builder = self.stream.launch_builder(&self.compute_hidden_layer_error_func);
        err_builder
            .arg(cur_layer.get_grads().get_data())
            .arg(next_e.get_data())
            .arg(next_w.get_data())
            .arg(cur_layer.get_outputs().get_data())
            .arg(cur_layer.get_predrop_outputs().get_data())
            .arg(cur_layer.get_preact_outputs().get_data())
            .arg(cur_layer.get_masks().get_data())
            .arg(&m_i32)
            .arg(&n_i32)
            .arg(&ec_i32)
            .arg(&act_mode_i32)
            .arg(&leaky_relu_coeff);

        let cfg = self.calculate_cfg2d(n, m, 0);

        unsafe {
            err_builder.launch(cfg).expect("Feed forward network back pass (hidden) layer function launch failed.");
        }

        self.gpu_backward_pass(cur_layer, optimiser, norm_optimiser, input, batch_size, lr, max_grad_norm, step);
    }
}