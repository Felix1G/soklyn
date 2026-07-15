use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::io::device::GpuContext;
use crate::{Activation, DenseBlock, LossFunc, Normalisation, Optimiser, Precision, PrecisionType, Regularisation};
use crate::core::{scramble_seed, Tensor2D};
use crate::log::Error;

impl GpuContext {
    pub(crate) fn gpu_forward_pass<T: PrecisionType>(
        &self,
        cur_layer: &DenseBlock<T>,
        input: &Tensor2D<T>,
        batch_size: usize,
        use_dropout: bool,
        step: usize,
    ) -> Result<(), Error> {
        self.check_tile_dim::<T>()?;

        let wc = cur_layer.get_weights().cols();
        let norm = cur_layer.get_normalisation();
        let act = cur_layer.get_activation();
        let mask_coeff = cur_layer.get_mask_coeff();
        let seed = scramble_seed(step as u32, cur_layer.get_weights().get_id() as u32);
        let use_bias = *norm != Normalisation::BatchNorm;

        let leaky_relu_coeff = match act {
            Activation::LeakyReLU(value) => *value,
            _ => 0.0,
        };

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
            .arg(if *norm == Normalisation::Disabled {
                cur_layer.get_preact_outputs().get_data()
            } else {
                cur_layer.get_prenorm_outputs().get_data()
            })
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
            builder = self.stream.launch_builder(match T::precision() {
                Precision::FP32 => &self.forward_pass_func1.0,
                Precision::FP16 => &self.forward_pass_func1.1,
            });
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

            let grid_x = if *norm == Normalisation::BatchNorm {
                wc
            } else {
                batch_size
            };

            let cfg = LaunchConfig {
                grid_dim: (grid_x as u32, 1, 1),
                block_dim: (self.tile_dim_2, 1, 1),
                shared_mem_bytes: self.tile_dim_2 * 4,
            };

            unsafe {
                builder.launch(cfg)?;
            }

            self.stream.synchronize()?;
        }

        builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.forward_pass_func2.0,
            Precision::FP16 => &self.forward_pass_func2.1,
        });
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
        &self,
        cur_layer: &DenseBlock<T>,
        target: &Tensor2D<T>,
        err_mode: LossFunc,
        activation: Activation,
    ) -> Result<(), Error> {
        self.gpu_broadcast(cur_layer.get_norm_weights_grad().get_data(), 0.0)?;
        self.gpu_broadcast(cur_layer.get_norm_biases_grad().get_data(), 0.0)?;

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
            let mut softmax_builder = self.stream.launch_builder(match T::precision() {
                Precision::FP32 => &self.softmax_func.0,
                Precision::FP16 => &self.softmax_func.1,
            });

            softmax_builder
                .arg(out_tensor.get_data())
                .arg(&m_u32)
                .arg(&cols_u32);

            let softmax_cfg = LaunchConfig {
                grid_dim: (m as u32, 1, 1),
                block_dim: (self.tile_dim_2, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                softmax_builder.launch(softmax_cfg)?;
            }

            self.stream.synchronize()?;
        }

        let mut err_builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.compute_output_layer_error_func.0,
            Precision::FP16 => &self.compute_output_layer_error_func.1,
        });
        err_builder
            .arg(out_tensor.get_data())
            .arg(cur_layer.get_preact_outputs().get_data())
            .arg(target.get_data());

        self.master_tensor(
            &mut err_builder,
            cur_layer.get_master_norm_weights(),
            cur_layer.get_norm_weights(),
        );

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

        unsafe {
            err_builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    pub(crate) fn gpu_backward_pass<T: PrecisionType>(
        &self,
        cur_layer: &DenseBlock<T>,
        optimiser: &Optimiser,
        norm_optimiser: &Optimiser,
        input: &Tensor2D<T>,
        batch_size: usize,
        lr: f32,
        max_grad_norm: f32,
        step: usize,
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

        let extract_optimiser_info = |optimiser: &Optimiser| match optimiser {
            Optimiser::SGD(b1, nest) => (*b1, 0.0, 0.0, if *nest { 1 } else { 0 }),
            Optimiser::Adam(b1, b2, eps) => (*b1, *b2, *eps, 0),
        };

        let (linear_beta1, linear_beta2, linear_epsilon, linear_nesterov) =
            extract_optimiser_info(optimiser);
        let (norm_beta1, norm_beta2, norm_epsilon, norm_nesterov) =
            extract_optimiser_info(norm_optimiser);

        let regu_coeff = match regularisation {
            Regularisation::None => 0.0,
            Regularisation::L1Regular(coeff) => *coeff,
            Regularisation::L2Regular(coeff) => *coeff,
        };

        let mut builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.backward_pass_func.0,
            Precision::FP16 => &self.backward_pass_func.1,
        });
        builder
            .arg(input.get_data())
            .arg(cur_layer.get_weights().get_data())
            .arg(cur_layer.get_biases().get_data());

        self.master_tensor(
            &mut builder,
            cur_layer.get_master_weights(),
            cur_layer.get_weights(),
        );
        self.master_tensor(
            &mut builder,
            cur_layer.get_master_biases(),
            cur_layer.get_biases(),
        );

        builder
            .arg(cur_layer.get_delta_prenorm_out().get_data())
            .arg(cur_layer.get_dv_weights().get_data())
            .arg(cur_layer.get_dv_biases().get_data())
            .arg(cur_layer.get_dm_weights().get_data())
            .arg(cur_layer.get_dm_biases().get_data())
            .arg(cur_layer.get_norm_weights().get_data())
            .arg(cur_layer.get_norm_biases().get_data());

        self.master_tensor(
            &mut builder,
            cur_layer.get_master_norm_weights(),
            cur_layer.get_norm_weights(),
        );
        self.master_tensor(
            &mut builder,
            cur_layer.get_master_norm_biases(),
            cur_layer.get_norm_biases(),
        );

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
        &self,
        cur_layer: &DenseBlock<T>,
        next_layer: &DenseBlock<T>,
        input: &Tensor2D<T>,
        optimiser: &Optimiser,
        norm_optimiser: &Optimiser,
        batch_size: usize,
        lr: f32,
        max_grad_norm: f32,
        activation: &Activation,
        step: usize,
    ) -> Result<(), Error> {
        self.check_tile_dim::<T>()?;
        self.gpu_broadcast(cur_layer.get_norm_weights_grad().get_data(), 0.0)?;
        self.gpu_broadcast(cur_layer.get_norm_biases_grad().get_data(), 0.0)?;

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

        let mut err_builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.compute_hidden_layer_error_func.0,
            Precision::FP16 => &self.compute_hidden_layer_error_func.1,
        });

        err_builder.arg(next_layer.get_delta_prenorm_out().get_data());

        self.master_tensor(
            &mut err_builder,
            next_layer.get_master_weights(),
            next_layer.get_weights(),
        );
        self.master_tensor(
            &mut err_builder,
            cur_layer.get_master_norm_weights(),
            cur_layer.get_norm_weights(),
        );

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

        let cfg = self.calculate_cfg2d(
            n,
            batch_size,
            match T::precision() {
                Precision::FP32 => 0,
                Precision::FP16 => 2 * self.tile_dim_2 * size_of::<T>() as u32,
            },
        );

        unsafe {
            err_builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        self.gpu_backward_pass(
            cur_layer,
            optimiser,
            norm_optimiser,
            input,
            batch_size,
            lr,
            max_grad_norm,
            step,
        )
    }
}