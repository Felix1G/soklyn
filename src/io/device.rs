use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchArgs, LaunchConfig, PushKernelArg};
use cudarc::driver::sys::CUctx_flags;
use cudarc::nvrtc::Ptx;
use crate::util::functions::{Activation, LossFunc, Normalisation, Optimiser, Regularisation};
use crate::layers::DenseBlock;
use crate::util::core::{scramble_seed, Tensor};
use crate::util::log::Error;
use crate::util::precision::{Precision, PrecisionType};

/// Carries the context of the connection to a GPU device.
#[allow(dead_code)]
pub struct GpuContext {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    cast_f32_f16_func: CudaFunction,
    cast_f16_f32_func: CudaFunction,
    broadcast_func: (CudaFunction, CudaFunction),
    gemm_func: (CudaFunction, CudaFunction), // A * B
    geam_func: (CudaFunction, CudaFunction), // A + B
    forward_pass_func0: (CudaFunction, CudaFunction), // Z = WX + B
    forward_pass_func1: (CudaFunction, CudaFunction), // Normalisation
    forward_pass_func2: (CudaFunction, CudaFunction), // Activation and dropout
    softmax_func: (CudaFunction, CudaFunction),
    compute_output_layer_error_func: (CudaFunction, CudaFunction),
    compute_hidden_layer_error_func: (CudaFunction, CudaFunction),
    backward_pass_func: (CudaFunction, CudaFunction),
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

        let cast_f32_f16 = load_kernel("cast_f32_f16_t_kernel");
        let cast_f16_f32 = load_kernel("cast_f16_f32_t_kernel");
        let broadcast_f32 = load_kernel("broadcast_kernel_f32");
        let gemm_f32 = load_kernel("sgemm_kernel");
        let geam_f32 = load_kernel("sgeam_kernel");
        let forward_pass_0_f32 = load_kernel("forward_pass_0_f32");
        let forward_pass_1_f32 = load_kernel("forward_pass_1_f32");
        let forward_pass_2_f32 = load_kernel("forward_pass_2_f32");
        let softmax_f32 = load_kernel("softmax_kernel_f32");
        let compute_output_layer_error_f32 = load_kernel("compute_output_layer_error_f32");
        let compute_hidden_layer_error_f32 = load_kernel("compute_hidden_layer_error_f32");
        let backward_pass_f32 = load_kernel("backward_pass_f32");
        let broadcast_f16 = load_kernel("broadcast_kernel_f16");
        let gemm_f16 = load_kernel("hgemm_kernel");
        let geam_f16 = load_kernel("hgeam_kernel");
        let forward_pass_0_f16 = load_kernel("forward_pass_0_f16");
        let forward_pass_1_f16 = load_kernel("forward_pass_1_f16");
        let forward_pass_2_f16 = load_kernel("forward_pass_2_f16");
        let softmax_f16 = load_kernel("softmax_kernel_f16");
        let compute_output_layer_error_f16 = load_kernel("compute_output_layer_error_f16");
        let compute_hidden_layer_error_f16 = load_kernel("compute_hidden_layer_error_f16");
        let backward_pass_f16 = load_kernel("backward_pass_f16");

        let stream = CudaContext::new_stream(&context).expect("Failed to create stream");
        context.set_flags(CUctx_flags::CU_CTX_SCHED_SPIN).expect("Failed to set context flags");

        let tile_dim_2 = tile_dim * tile_dim;

        Self {
            context,
            stream,
            cast_f32_f16_func: cast_f32_f16,
            cast_f16_f32_func: cast_f16_f32,
            broadcast_func: (broadcast_f32, broadcast_f16),
            gemm_func: (gemm_f32, gemm_f16),
            geam_func: (geam_f32, geam_f16),
            forward_pass_func0: (forward_pass_0_f32, forward_pass_0_f16),
            forward_pass_func1: (forward_pass_1_f32, forward_pass_1_f16),
            forward_pass_func2: (forward_pass_2_f32, forward_pass_2_f16),
            softmax_func: (softmax_f32, softmax_f16),
            compute_output_layer_error_func: (compute_output_layer_error_f32, compute_output_layer_error_f16),
            compute_hidden_layer_error_func: (compute_hidden_layer_error_f32, compute_hidden_layer_error_f16),
            backward_pass_func: (backward_pass_f32, backward_pass_f16),
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

    fn check_tile_dim<T: PrecisionType>(&self) -> Result<(), Error>{
        if T::precision() == Precision::FP16 && self.tile_dim != 16 {
            return Err(Error::HardwareConstraintViolation {
                precision: Precision::FP16,
                expected: 16,
                found: self.tile_dim,
            });
        }

        Ok(())
    }

    fn master_tensor<'a, T: PrecisionType>(&self, builder: &mut LaunchArgs<'a>, tensor: &'a Option<Tensor<f32>>, fallback: &'a Tensor<T>) {
        if let Some(tensor_data) = tensor {
            builder.arg(tensor_data.get_data());
        } else {
            builder.arg(fallback.get_data());
        }
    }

    pub(crate) fn gpu_cast_t<T: PrecisionType, U: PrecisionType>(
        &self, src: &CudaSlice<T>, dst: &CudaSlice<U>
    ) -> Result<(), Error> {
        if src.len() != dst.len() {
            return Err(Error::MismatchedDimensions {
                context: "GPU precision cast source vs destination buffer sizes",
                expected: src.len(),
                found: dst.len(),
            });
        }

        let len = src.len() as u32;

        let mut builder;
        if T::precision() == Precision::FP32 && U::precision() == Precision::FP16 {
            builder = self.stream.launch_builder(&self.cast_f32_f16_func);
        } else if T::precision() == Precision::FP16 && U::precision() == Precision::FP32 {
            builder = self.stream.launch_builder(&self.cast_f16_f32_func);
        } else {
            return Err(Error::UnsupportedTypeCast {
                from: T::precision(),
                to: U::precision(),
            });
        }

        builder.arg(src).arg(dst).arg(&len);

        let cfg = LaunchConfig {
            grid_dim: ((len + self.tile_dim_2_minus_1) / self.tile_dim_2, 1, 1),
            block_dim: (self.tile_dim_2, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe { builder.launch(cfg)?; }

        Ok(())
    }

    pub(crate) fn gpu_broadcast<T: PrecisionType>(&self, dst: &CudaSlice<T>, v: T) {
        let len = dst.len();
        let len_u32 = len as u32;

        let mut builder = self.stream.launch_builder(
            match T::precision() {
                Precision::FP32 => &self.broadcast_func.0,
                Precision::FP16 => &self.broadcast_func.1,
            }
        );
        builder.arg(dst).arg(&v).arg(&len_u32);

        let cfg = LaunchConfig {
            grid_dim: ((len as u32 + self.tile_dim_2_minus_1) / self.tile_dim_2, 1, 1),
            block_dim: (self.tile_dim_2, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            builder.launch(cfg).expect("broadcast launch failed.");
        }
    }

    // Single-precision General Matrix Multiply
    pub(crate) fn gpu_matrix_mul<T: PrecisionType>(&self,
                                 a_dev: &CudaSlice<T>, m: usize, n: usize,
                                 b_dev: &CudaSlice<T>, p: usize,
                                 c_dev: &mut CudaSlice<T>) {
        let mut builder = self.stream.launch_builder(
            match T::precision() {
                Precision::FP32 => &self.gemm_func.0,
                Precision::FP16 => &self.gemm_func.1,
            }
        );
        builder.arg(a_dev).arg(b_dev).arg(c_dev);
        builder.arg(&m).arg(&n).arg(&p);
        builder.arg(&self.tile_dim);

        let cfg = self.calculate_cfg2d(p, m, 2 * self.tile_dim_2 * size_of::<T>() as u32);

        unsafe {
            builder.launch(cfg).expect("matrix multiplication launch failed.");
        }

        self.stream.synchronize().unwrap();
    }

    pub(crate) fn gpu_matrix_add<T: PrecisionType>(&self,
                                 a_dev: &CudaSlice<T>, m: usize, n: usize,
                                 b_dev: &CudaSlice<T>,
                                 c_dev: &mut CudaSlice<T>) {
        let mut builder = self.stream.launch_builder(
            match T::precision() {
                Precision::FP32 => &self.geam_func.0,
                Precision::FP16 => &self.geam_func.1,
            }
        );
        builder.arg(a_dev).arg(b_dev).arg(c_dev);
        builder.arg(&m).arg(&n);

        let cfg = self.calculate_cfg2d(n, m, 0);

        unsafe {
            builder.launch(cfg).expect("matrix addition launch failed.");
        }

        self.stream.synchronize().unwrap();
    }

    pub(crate) fn gpu_forward_pass<T: PrecisionType>(
        &self, cur_layer: &DenseBlock<T>, input: &Tensor<T>, batch_size: usize,
        leaky_relu_coeff: f32, use_dropout: bool, step: usize
    ) -> Result<(), Error> {
        self.check_tile_dim::<T>()?;

        let wc = cur_layer.get_weights().cols();
        let norm = cur_layer.get_normalisation();
        let act = cur_layer.get_activation();
        let mask_coeff = cur_layer.get_mask_coeff();
        let seed = scramble_seed(step as u32, cur_layer.get_weights().get_id() as u32);
        let use_bias = *norm != Normalisation::BatchNorm;

        let use_bias_u32 = use_bias as u32;
        let m_u32 = batch_size as u32;
        let n_u32 = input.cols() as u32;
        let wc_u32 = wc as u32;
        let norm_mode_u32 = norm.ordinal() as u32;
        let act_mode_u32 = act.ordinal() as u32;
        let use_dropout_u32 = use_dropout as u32;

        let mut builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.forward_pass_func0.0,
            Precision::FP16 => &self.forward_pass_func0.1,
        });

        builder
            .arg(
                if *norm == Normalisation::Disabled { cur_layer.get_preact_outputs().get_data() }
                else { cur_layer.get_prenorm_outputs().get_data() }
            )
            .arg(input.get_data())
            .arg(cur_layer.get_weights().get_data())
            .arg(cur_layer.get_biases().get_data())
            .arg(&use_bias_u32)
            .arg(&m_u32)
            .arg(&n_u32)
            .arg(&wc_u32)
            .arg(&self.tile_dim);

        let cfg = self.calculate_cfg2d(wc, batch_size, 2 * self.tile_dim_2 * size_of::<T>() as u32);

        unsafe {
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        // Normalisation
        if *norm != Normalisation::Disabled {
            builder = self.stream.launch_builder(
                match T::precision() {
                    Precision::FP32 => &self.forward_pass_func1.0,
                    Precision::FP16 => &self.forward_pass_func1.1,
                }
            );
            builder
                .arg(cur_layer.get_preact_outputs().get_data())
                .arg(cur_layer.get_centered_outputs().get_data())
                .arg(cur_layer.get_prenorm_outputs().get_data())
                .arg(cur_layer.get_norm_weights().get_data())
                .arg(cur_layer.get_norm_biases().get_data())
                .arg(cur_layer.get_norm_rstd().get_data())
                .arg(&m_u32)
                .arg(&wc_u32)
                .arg(&norm_mode_u32);

            let grid_x = if *norm == Normalisation::BatchNorm { wc } else { batch_size };

            let cfg = LaunchConfig {
                grid_dim: (grid_x as u32, 1, 1),
                block_dim: (self.tile_dim_2, 1, 1),
                shared_mem_bytes: self.tile_dim_2 * 4
            };

            unsafe {
                builder.launch(cfg)?;
            }

            self.stream.synchronize()?;
        }

        builder = self.stream.launch_builder(
            match T::precision() {
                Precision::FP32 => &self.forward_pass_func2.0,
                Precision::FP16 => &self.forward_pass_func2.1,
            }
        );
        builder
            .arg(cur_layer.get_outputs().get_data())
            .arg(cur_layer.get_predrop_outputs().get_data())
            .arg(cur_layer.get_preact_outputs().get_data())
            .arg(cur_layer.get_masks().get_data())
            .arg(&m_u32)
            .arg(&wc_u32)
            .arg(&act_mode_u32)
            .arg(&leaky_relu_coeff)
            .arg(&use_dropout_u32)
            .arg(&mask_coeff)
            .arg(&seed);

        let cfg = self.calculate_cfg2d(wc, batch_size, 0);

        unsafe {
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    pub(crate) fn gpu_compute_output_layer_error<T: PrecisionType>(
        &self, cur_layer: &DenseBlock<T>, target: &Tensor<T>,
        err_mode: LossFunc, activation: Activation
    ) -> Result<(), Error> {
        self.gpu_broadcast(cur_layer.get_norm_weights_grad().get_data(), 0.0);
        self.gpu_broadcast(cur_layer.get_norm_biases_grad().get_data(), 0.0);

        let out_tensor = cur_layer.get_outputs();
        let m = out_tensor.rows();
        let n = cur_layer.get_weights().cols();

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let err_mode_u32 = err_mode as u32;
        let norm_mode_u32 = cur_layer.get_normalisation().ordinal() as u32;
        let act_mode_u32 = activation.ordinal() as u32;

        let leaky_relu_coeff = match activation {
            Activation::LeakyReLU(value) => value,
            _ => 0.0,
        };

        // Run softmax if needed first
        if activation == Activation::Softmax {
            let cols_u32 = out_tensor.cols() as u32;
            let mut softmax_builder = self.stream.launch_builder(
                match T::precision() {
                    Precision::FP32 => &self.softmax_func.0,
                    Precision::FP16 => &self.softmax_func.1,
                }
            );

            softmax_builder
                .arg(out_tensor.get_data())
                .arg(&m_u32)
                .arg(&cols_u32);

            let softmax_cfg = LaunchConfig {
                grid_dim: (m as u32, 1, 1),
                block_dim: (self.tile_dim_2, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe { softmax_builder.launch(softmax_cfg)?; }

            self.stream.synchronize()?;
        }

        let mut err_builder = self.stream.launch_builder(
            match T::precision() {
                Precision::FP32 => &self.compute_output_layer_error_func.0,
                Precision::FP16 => &self.compute_output_layer_error_func.1,
            }
        );
        err_builder
            .arg(out_tensor.get_data())
            .arg(cur_layer.get_preact_outputs().get_data())
            .arg(target.get_data());

        self.master_tensor(&mut err_builder, cur_layer.get_master_norm_weights(), cur_layer.get_norm_weights());

        err_builder
            .arg(cur_layer.get_grads().get_data())
            .arg(cur_layer.get_delta_prenorm_out().get_data())
            .arg(cur_layer.get_norm_weights_grad().get_data())
            .arg(cur_layer.get_norm_biases_grad().get_data())
            .arg(cur_layer.get_norm_rstd().get_data())
            .arg(cur_layer.get_centered_outputs().get_data())
            .arg(cur_layer.get_prenorm_outputs().get_data())
            .arg(&m_u32)
            .arg(&n_u32)
            .arg(&err_mode_u32)
            .arg(&norm_mode_u32)
            .arg(&act_mode_u32)
            .arg(&leaky_relu_coeff);

        let cfg = self.calculate_cfg2d(n, m, 0);

        unsafe { err_builder.launch(cfg)?; }

        self.stream.synchronize()?;

        Ok(())
    }

    pub(crate) fn gpu_backward_pass<T: PrecisionType>(
        &self, cur_layer: &DenseBlock<T>,
        optimiser: &Optimiser, norm_optimiser: &Optimiser, input: &Tensor<T>,
        batch_size: usize, lr: f32, max_grad_norm: f32, step: usize
    ) -> Result<(), Error> {
        let n = cur_layer.get_weights().cols();
        let wr = cur_layer.get_weights().rows();
        let norm = cur_layer.get_normalisation();
        let regularisation = cur_layer.get_regularisation();
        let use_bias = *norm != Normalisation::BatchNorm;

        let use_bias_u32 = use_bias as u32;
        let norm_mode_u32 = norm.ordinal() as u32;
        let m_u32 = batch_size as u32;
        let n_u32 = n as u32;
        let wr_u32 = wr as u32;
        let optimiser_u32 = optimiser.ordinal() as u32;
        let norm_optimiser_u32 = norm_optimiser.ordinal() as u32;
        let regu_mode_u32 = regularisation.ordinal() as u32;
        let step_u32 = step as u32;

        let extract_optimiser_info = |optimiser: &Optimiser| {
            match optimiser {
                Optimiser::SGD(b1, nest) => {
                    (*b1, 0.0, 0.0, if *nest { 1 } else { 0 })
                }
                Optimiser::Adam(b1, b2, eps) => {
                    (*b1, *b2, *eps, 0)
                }
            }
        };

        let (linear_beta1, linear_beta2, linear_epsilon, linear_nesterov) = extract_optimiser_info(optimiser);
        let (norm_beta1, norm_beta2, norm_epsilon, norm_nesterov) = extract_optimiser_info(norm_optimiser);

        let regu_coeff = match regularisation {
            Regularisation::None => 0.0,
            Regularisation::L1Regular(coeff) => *coeff,
            Regularisation::L2Regular(coeff) => *coeff,
        };

        let mut builder = self.stream.launch_builder(
            match T::precision() {
                Precision::FP32 => &self.backward_pass_func.0,
                Precision::FP16 => &self.backward_pass_func.1,
            }
        );
        builder
            .arg(input.get_data())
            .arg(cur_layer.get_weights().get_data())
            .arg(cur_layer.get_biases().get_data());

        self.master_tensor(&mut builder, cur_layer.get_master_weights(), cur_layer.get_weights());
        self.master_tensor(&mut builder, cur_layer.get_master_biases(), cur_layer.get_biases());

        builder
            .arg(cur_layer.get_delta_prenorm_out().get_data())
            .arg(cur_layer.get_dv_weights().get_data())
            .arg(cur_layer.get_dv_biases().get_data())
            .arg(cur_layer.get_dm_weights().get_data())
            .arg(cur_layer.get_dm_biases().get_data())
            .arg(cur_layer.get_norm_weights().get_data())
            .arg(cur_layer.get_norm_biases().get_data());

        self.master_tensor(&mut builder, cur_layer.get_master_norm_weights(), cur_layer.get_norm_weights());
        self.master_tensor(&mut builder, cur_layer.get_master_norm_biases(), cur_layer.get_norm_biases());

        builder
            .arg(cur_layer.get_norm_weights_grad().get_data())
            .arg(cur_layer.get_norm_biases_grad().get_data())
            .arg(cur_layer.get_dv_norm_weights().get_data())
            .arg(cur_layer.get_dv_norm_biases().get_data())
            .arg(cur_layer.get_dm_norm_weights().get_data())
            .arg(cur_layer.get_dm_norm_biases().get_data())
            .arg(&use_bias_u32)
            .arg(&norm_mode_u32)
            .arg(&m_u32)
            .arg(&n_u32)
            .arg(&wr_u32)
            .arg(&lr)
            .arg(&max_grad_norm)
            .arg(&optimiser_u32)
            .arg(&norm_optimiser_u32)
            .arg(&linear_beta1)
            .arg(&linear_beta2)
            .arg(&linear_epsilon)
            .arg(&linear_nesterov)
            .arg(&norm_beta1)
            .arg(&norm_beta2)
            .arg(&norm_epsilon)
            .arg(&norm_nesterov)
            .arg(&regu_mode_u32)
            .arg(&regu_coeff)
            .arg(&step_u32);

        let cfg = self.calculate_cfg2d(n, wr, 0);
        
        unsafe {
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    pub(crate) fn gpu_hidden_layer_backward_pass<T: PrecisionType>(
        &self, cur_layer: &DenseBlock<T>, next_layer: &DenseBlock<T>, input: &Tensor<T>,
        optimiser: &Optimiser, norm_optimiser: &Optimiser,
        batch_size: usize, lr: f32, max_grad_norm: f32,
        activation: &Activation, step: usize
    ) -> Result<(), Error> {
        self.check_tile_dim::<T>()?;
        self.gpu_broadcast(cur_layer.get_norm_weights_grad().get_data(), 0.0);
        self.gpu_broadcast(cur_layer.get_norm_biases_grad().get_data(), 0.0);

        let n = cur_layer.get_weights().cols();
        let ec = next_layer.get_outputs().cols();

        let m_u32 = batch_size as u32;
        let n_u32 = n as u32;
        let ec_u32 = ec as u32;
        let norm_mode_u32 = cur_layer.get_normalisation().ordinal() as u32;
        let act_mode_u32 = activation.ordinal() as u32;

        let leaky_relu_coeff = match activation {
            Activation::LeakyReLU(value) => *value,
            _ => 0.0,
        };

        let mut err_builder = self.stream.launch_builder(
            match T::precision() {
                Precision::FP32 => &self.compute_hidden_layer_error_func.0,
                Precision::FP16 => &self.compute_hidden_layer_error_func.1,
            }
        );

        err_builder.arg(next_layer.get_delta_prenorm_out().get_data());

        self.master_tensor(&mut err_builder, next_layer.get_master_weights(), next_layer.get_weights());
        self.master_tensor(&mut err_builder, cur_layer.get_master_norm_weights(), cur_layer.get_norm_weights());

        err_builder
            .arg(cur_layer.get_grads().get_data())
            .arg(cur_layer.get_delta_prenorm_out().get_data())
            .arg(cur_layer.get_norm_weights_grad().get_data())
            .arg(cur_layer.get_norm_biases_grad().get_data())
            .arg(cur_layer.get_norm_rstd().get_data())
            .arg(cur_layer.get_centered_outputs().get_data())
            .arg(cur_layer.get_prenorm_outputs().get_data())
            .arg(cur_layer.get_predrop_outputs().get_data())
            .arg(cur_layer.get_preact_outputs().get_data())
            .arg(cur_layer.get_masks().get_data())
            .arg(&m_u32)
            .arg(&n_u32)
            .arg(&ec_u32)
            .arg(&norm_mode_u32)
            .arg(&act_mode_u32)
            .arg(&leaky_relu_coeff);

        let cfg = self.calculate_cfg2d(n, batch_size, match T::precision() {
            Precision::FP32 => 0,
            Precision::FP16 => 2 * self.tile_dim_2 * size_of::<T>() as u32,
        });

        unsafe { err_builder.launch(cfg)?; }

        self.stream.synchronize()?;

        self.gpu_backward_pass(cur_layer, optimiser, norm_optimiser, input, batch_size, lr, max_grad_norm, step)
    }
}