use crate::TrainContext;
use crate::core::{scramble_seed, Tensor2D};
use crate::io::device::GpuContext;
use crate::log::Error;
use crate::{
    Activation, DenseBlock, LossFunc, Normalisation, Optimiser, Precision, PrecisionType,
    Regularisation,
};
use cudarc::driver::{LaunchConfig, PushKernelArg};

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
        let seed = scramble_seed(
            u32::try_from(step)?,
            u32::try_from(cur_layer.get_weights().get_id())?,
        );
        let use_bias = *norm != Normalisation::BatchNorm;

        let leaky_relu_coeff = match act {
            Activation::LeakyReLU { coeff } => *coeff,
            _ => 0.0,
        };

        let use_bias_u32 = u32::from(use_bias);
        let m_u32 = u32::try_from(batch_size)?;
        let n_u32 = u32::try_from(input.cols())?;
        let wc_u32 = u32::try_from(wc)?;
        let norm_mode_u32 = u32::try_from(norm.ordinal())?;
        let act_mode_u32 = u32::try_from(act.ordinal())?;
        let use_dropout_u32 = u32::from(use_dropout);

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

        let cfg = self.calculate_cfg2d(
            wc,
            batch_size,
            2 * self.tile_dim_2 * u32::try_from(size_of::<T>())?,
        )?;

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
                grid_dim: (u32::try_from(grid_x)?, 1, 1),
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

        let cfg = self.calculate_cfg2d(wc, batch_size, 0)?;

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

        let m_u32 = u32::try_from(m)?;
        let n_u32 = u32::try_from(n)?;
        let err_mode_u32 = err_mode as u32;
        let norm_mode_u32 = u32::try_from(cur_layer.get_normalisation().ordinal())?;
        let act_mode_u32 = u32::try_from(activation.ordinal())?;

        let leaky_relu_coeff = match activation {
            Activation::LeakyReLU { coeff } => coeff,
            _ => 0.0,
        };

        // Run softmax if needed first
        if activation == Activation::Softmax {
            let cols_u32 = u32::try_from(out_tensor.cols())?;
            let mut softmax_builder = self.stream.launch_builder(match T::precision() {
                Precision::FP32 => &self.softmax_func.0,
                Precision::FP16 => &self.softmax_func.1,
            });

            softmax_builder
                .arg(out_tensor.get_data())
                .arg(&m_u32)
                .arg(&cols_u32);

            let softmax_cfg = LaunchConfig {
                grid_dim: (u32::try_from(m)?, 1, 1),
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

        Self::master_tensor(
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

        let cfg = self.calculate_cfg2d(n, m, 0)?;

        unsafe {
            err_builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    pub(crate) fn gpu_backward_pass<T: PrecisionType>(
        &self,
        cur_layer: &DenseBlock<T>,
        input: &Tensor2D<T>,
        train_ctx: &TrainContext,
        step: usize,
    ) -> Result<(), Error> {
        let n = cur_layer.get_weights().cols();
        let wr = cur_layer.get_weights().rows();
        let norm = cur_layer.get_normalisation();
        let regularisation = cur_layer.get_regularisation();
        let use_bias = *norm != Normalisation::BatchNorm;

        let use_bias_u32 = u32::from(use_bias);
        let norm_mode_u32 = u32::try_from(norm.ordinal())?;
        let m_u32 = u32::try_from(train_ctx.batch_size)?;
        let n_u32 = u32::try_from(n)?;
        let wr_u32 = u32::try_from(wr)?;
        let optimiser_u32 = u32::try_from(train_ctx.optimiser.ordinal())?;
        let norm_optimiser_u32 = u32::try_from(train_ctx.norm_optimiser.ordinal())?;
        let regu_mode_u32 = u32::try_from(regularisation.ordinal())?;
        let step_u32 = u32::try_from(step)?;

        let extract_optimiser_info = |optimiser: &Optimiser| match optimiser {
            Optimiser::SGD {
                v_coeff: b1,
                nesterov,
            } => (*b1, 0.0, 0.0, i32::from(*nesterov)),
            Optimiser::Adam {
                m_coeff: b1,
                v_coeff: b2,
                epsilon,
            } => (*b1, *b2, *epsilon, 0),
        };

        let (linear_beta1, linear_beta2, linear_epsilon, linear_nesterov) =
            extract_optimiser_info(train_ctx.optimiser);
        let (norm_beta1, norm_beta2, norm_epsilon, norm_nesterov) =
            extract_optimiser_info(train_ctx.norm_optimiser);

        let regu_coeff = match regularisation {
            Regularisation::None => 0.0,
            Regularisation::L1Regular { regu_coeff } | Regularisation::L2Regular { regu_coeff } => {
                *regu_coeff
            }
        };

        let mut builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.backward_pass_func.0,
            Precision::FP16 => &self.backward_pass_func.1,
        });
        builder
            .arg(input.get_data())
            .arg(cur_layer.get_weights().get_data())
            .arg(cur_layer.get_biases().get_data());

        Self::master_tensor(
            &mut builder,
            cur_layer.get_master_weights(),
            cur_layer.get_weights(),
        );
        Self::master_tensor(
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

        Self::master_tensor(
            &mut builder,
            cur_layer.get_master_norm_weights(),
            cur_layer.get_norm_weights(),
        );
        Self::master_tensor(
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
            .arg(&train_ctx.learn_rate)
            .arg(&train_ctx.grad_clamp)
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

        let cfg = self.calculate_cfg2d(n, wr, 0)?;

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
        activation: Activation,
        train_ctx: &TrainContext,
        step: usize,
    ) -> Result<(), Error> {
        self.check_tile_dim::<T>()?;
        self.gpu_broadcast(cur_layer.get_norm_weights_grad().get_data(), 0.0)?;
        self.gpu_broadcast(cur_layer.get_norm_biases_grad().get_data(), 0.0)?;

        let n = cur_layer.get_weights().cols();
        let ec = next_layer.get_outputs().cols();

        let m_u32 = u32::try_from(train_ctx.batch_size)?;
        let n_u32 = u32::try_from(n)?;
        let ec_u32 = u32::try_from(ec)?;
        let norm_mode_u32 = u32::try_from(cur_layer.get_normalisation().ordinal())?;
        let act_mode_u32 = u32::try_from(activation.ordinal())?;

        let leaky_relu_coeff = match activation {
            Activation::LeakyReLU { coeff } => coeff,
            _ => 0.0,
        };

        let mut err_builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.compute_hidden_layer_error_func.0,
            Precision::FP16 => &self.compute_hidden_layer_error_func.1,
        });

        err_builder.arg(next_layer.get_delta_prenorm_out().get_data());

        Self::master_tensor(
            &mut err_builder,
            next_layer.get_master_weights(),
            next_layer.get_weights(),
        );
        Self::master_tensor(
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
            train_ctx.batch_size,
            match T::precision() {
                Precision::FP32 => 0,
                Precision::FP16 => 2 * self.tile_dim_2 * u32::try_from(size_of::<T>())?,
            },
        )?;

        unsafe {
            err_builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        self.gpu_backward_pass(cur_layer, input, train_ctx, step)
    }
}
