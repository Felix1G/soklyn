use crate::core::{scramble_seed, Tensor4D};
use crate::io::device::GpuContext;
use crate::log::Error;
use crate::{
    Activation, Complex32, ConvAlgorithm, ConvBlock, LossFunc, Normalisation, Precision,
    PrecisionType,
};
use cudarc::driver::{LaunchConfig, PushKernelArg};

impl GpuContext {
    /// For `auto_pad`, it is incorporated into `ow` and `oh` such that the width/height purposely
    /// overflows the input image to apply padding (`ow` and `oh` are thread indices).
    /// The value of the width/height is calculated inside [`KernelConfig`].
    /// The kernel code simply calculates the padded values from overflowing index.
    ///
    /// The kernel parameter `pad` is just an alias for the offset of the starting index coordinates i.e. `(-pad, -pad)`.
    pub(crate) fn gpu_conv_forward_pass<T: PrecisionType>(
        &self,
        cur_layer: &ConvBlock<T>,
        input: &Tensor4D<T>,
        fft_output_size: &(usize, usize),
        batch_size: usize,
        use_dropout: bool,
        step: usize,
    ) -> Result<(), Error> {
        let norm = cur_layer.get_normalisation();
        let act = cur_layer.get_activation();
        let p_cfg = cur_layer.get_pooling_cfg();

        let allow_pass_1 = *norm != Normalisation::Disabled;
        let allow_pass_2 = *act != Activation::Identity || p_cfg.is_enabled() || use_dropout;

        let output = if allow_pass_1 {
            &cur_layer.get_norm_cache().unwrap().prenorm_features
        } else if allow_pass_2 {
            cur_layer.get_preact_features()
        } else {
            cur_layer.get_features()
        };

        match cur_layer.get_algorithm() {
            ConvAlgorithm::Spatial => self.gpu_conv_spatial(cur_layer, input, output)?,
            ConvAlgorithm::FrequencyFFT => {
                self.gpu_conv_fft(cur_layer, input, output, fft_output_size)?;
            }
        }

        if allow_pass_1 {
            self.gpu_conv_norm_pass(cur_layer, input, batch_size, allow_pass_2)?;
        }

        if allow_pass_2 {
            self.gpu_conv_activation_pool_dropout_pass(cur_layer, input, use_dropout, step)?;
        }

        Ok(())
    }

    fn gpu_conv_spatial<T: PrecisionType>(
        &self,
        cur_layer: &ConvBlock<T>,
        input: &Tensor4D<T>,
        output: &Tensor4D<T>,
    ) -> Result<(), Error> {
        let f_cfg = cur_layer.get_filter_cfg();
        let f_stride = f_cfg.get_stride();
        let f_dilation = f_cfg.get_dilation();
        let norm = cur_layer.get_normalisation();
        let use_bias = *norm != Normalisation::BatchNorm;
        let filter_weights = cur_layer.get_filter_weights();

        let use_bias_u32 = u32::from(use_bias);
        let n_u32 = u32::try_from(input.batches())?;
        let ic_u32 = u32::try_from(input.channels())?;
        let iw_u32 = u32::try_from(input.width())?;
        let ih_u32 = u32::try_from(input.height())?;

        let o_size = f_cfg.elements_from_length(&(input.width(), input.height()));
        let ow_u32 = u32::try_from(o_size.0)?;
        let oh_u32 = u32::try_from(o_size.1)?;
        let oc_u32 = u32::try_from(output.channels())?;

        let fw_u32 = u32::try_from(filter_weights.width())?;
        let fh_u32 = u32::try_from(filter_weights.height())?;
        let f_pad_u32 = u32::try_from(f_cfg.get_pad())?;
        let f_pad_mode_u32 = u32::try_from(f_cfg.get_pad_type().ordinal())?;
        let f_stride_x_u32 = u32::try_from(f_stride.0)?;
        let f_stride_y_u32 = u32::try_from(f_stride.1)?;
        let f_dilation_x_u32 = u32::try_from(f_dilation.0)?;
        let f_dilation_y_u32 = u32::try_from(f_dilation.1)?;

        let mut builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.conv_forward_pass_func0.0,
            Precision::FP16 => &self.conv_forward_pass_func0.1,
        });

        builder
            .arg(output.get_data())
            .arg(input.get_data())
            .arg(filter_weights.get_data())
            .arg(cur_layer.get_filter_biases().get_data())
            .arg(&use_bias_u32)
            .arg(&ic_u32)
            .arg(&oc_u32)
            .arg(&iw_u32)
            .arg(&ih_u32)
            .arg(&ow_u32)
            .arg(&oh_u32)
            .arg(&fw_u32)
            .arg(&fh_u32)
            .arg(&f_pad_u32)
            .arg(&f_pad_mode_u32)
            .arg(&f_stride_x_u32)
            .arg(&f_stride_y_u32)
            .arg(&f_dilation_x_u32)
            .arg(&f_dilation_y_u32);

        let tile_h = (self.tile_dim - 1) * u32::try_from(f_stride.1)?
            + u32::try_from(f_cfg.actual_height())?;
        let tile_w =
            (self.tile_dim - 1) * u32::try_from(f_stride.0)? + u32::try_from(f_cfg.actual_width())?;

        let cfg = self.calculate_cfg4d(
            n_u32 as usize,
            oc_u32 as usize,
            oh_u32 as usize,
            ow_u32 as usize,
            tile_w * tile_h * u32::try_from(size_of::<T>())?,
        )?;

        unsafe {
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn gpu_conv_fft<T: PrecisionType>(
        &self,
        cur_layer: &ConvBlock<T>,
        input: &Tensor4D<T>,
        output: &Tensor4D<T>,
        fft_output_size: &(usize, usize),
    ) -> Result<(), Error> {
        let f_cfg = cur_layer.get_filter_cfg();
        let f_stride = f_cfg.get_stride();
        let f_dilation = f_cfg.get_dilation();
        let norm = cur_layer.get_normalisation();
        let use_bias = *norm != Normalisation::BatchNorm;
        let filter_weights = cur_layer.get_filter_weights();

        let use_bias_u32 = u32::from(use_bias);
        let n_u32 = u32::try_from(input.batches())?;
        let ic_u32 = u32::try_from(input.channels())?;
        let iw_u32 = u32::try_from(input.width())?;
        let ih_u32 = u32::try_from(input.height())?;

        let o_size = f_cfg.elements_from_length(&(input.width(), input.height()));
        let ow_u32 = u32::try_from(o_size.0)?;
        let oh_u32 = u32::try_from(o_size.1)?;
        let oc_u32 = u32::try_from(output.channels())?;

        let fw_u32 = u32::try_from(filter_weights.width())?;
        let fh_u32 = u32::try_from(filter_weights.height())?;
        let f_pad_u32 = u32::try_from(f_cfg.get_pad())?;
        let f_pad_mode_u32 = u32::try_from(f_cfg.get_pad_type().ordinal())?;
        let f_stride_x_u32 = u32::try_from(f_stride.0)?;
        let f_stride_y_u32 = u32::try_from(f_stride.1)?;
        let f_dilation_x_u32 = u32::try_from(f_dilation.0)?;
        let f_dilation_y_u32 = u32::try_from(f_dilation.1)?;

        let fft_ow_u32 = u32::try_from(fft_output_size.0)?;
        let fft_oh_u32 = u32::try_from(fft_output_size.1)?;

        let mut row_builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.conv_fft_row_transform_func.0,
            Precision::FP16 => &self.conv_fft_row_transform_func.1,
        });

        row_builder
            .arg(cur_layer.get_fft_input().unwrap())
            .arg(cur_layer.get_fft_weights().unwrap())
            .arg(cur_layer.get_twiddle_lut_width().unwrap())
            .arg(input.get_data())
            .arg(filter_weights.get_data())
            .arg(&n_u32)
            .arg(&ic_u32)
            .arg(&iw_u32)
            .arg(&ih_u32)
            .arg(&fft_ow_u32)
            .arg(&fft_oh_u32)
            .arg(&fw_u32)
            .arg(&fh_u32)
            .arg(&f_pad_u32)
            .arg(&f_pad_mode_u32)
            .arg(&f_dilation_x_u32)
            .arg(&f_dilation_y_u32);

        let cfg = LaunchConfig {
            grid_dim: ((n_u32 + oc_u32) * ic_u32 * fft_oh_u32, 1, 1),
            block_dim: (self.tile_dim_2, 1, 1),
            shared_mem_bytes: fft_ow_u32 * u32::try_from(size_of::<Complex32>())?,
        };

        unsafe {
            row_builder.launch(cfg)?;
        }

        let mut col_builder = self
            .stream
            .launch_builder(&self.conv_fft_col_transform_func);

        col_builder
            .arg(cur_layer.get_fft_input().unwrap())
            .arg(cur_layer.get_fft_weights().unwrap())
            .arg(cur_layer.get_twiddle_lut_height().unwrap())
            .arg(&n_u32)
            .arg(&ic_u32)
            .arg(&fft_ow_u32)
            .arg(&fft_oh_u32);

        let cfg = LaunchConfig {
            grid_dim: ((n_u32 + oc_u32) * ic_u32 * fft_ow_u32, 1, 1),
            block_dim: (self.tile_dim_2, 1, 1),
            shared_mem_bytes: fft_oh_u32 * u32::try_from(size_of::<Complex32>())?,
        };

        unsafe {
            col_builder.launch(cfg)?;
        }

        let mut ifft_row_builder = self
            .stream
            .launch_builder(&self.conv_elem_mul_ifft_row_func);

        ifft_row_builder
            .arg(cur_layer.get_fft_input().unwrap())
            .arg(cur_layer.get_fft_weights().unwrap())
            .arg(cur_layer.get_fft_output().unwrap())
            .arg(cur_layer.get_twiddle_lut_width().unwrap())
            .arg(&oc_u32)
            .arg(&ic_u32)
            .arg(&fft_ow_u32)
            .arg(&fft_oh_u32);

        let cfg = LaunchConfig {
            grid_dim: (n_u32 * oc_u32 * fft_oh_u32, 1, 1),
            block_dim: (self.tile_dim_2, 1, 1),
            shared_mem_bytes: fft_ow_u32 * u32::try_from(size_of::<Complex32>())?,
        };

        unsafe {
            ifft_row_builder.launch(cfg)?;
        }

        let mut ifft_col_builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.conv_ifft_col_transform_func.0,
            Precision::FP16 => &self.conv_ifft_col_transform_func.1,
        });

        ifft_col_builder
            .arg(output.get_data())
            .arg(cur_layer.get_fft_output().unwrap())
            .arg(cur_layer.get_twiddle_lut_height().unwrap())
            .arg(cur_layer.get_filter_biases().get_data())
            .arg(&use_bias_u32)
            .arg(&oc_u32)
            .arg(&fft_ow_u32)
            .arg(&fft_oh_u32)
            .arg(&ow_u32)
            .arg(&oh_u32)
            .arg(&f_stride_x_u32)
            .arg(&f_stride_y_u32);

        let cfg = LaunchConfig {
            grid_dim: (n_u32 * oc_u32 * ow_u32, 1, 1),
            block_dim: (self.tile_dim_2, 1, 1),
            shared_mem_bytes: fft_oh_u32 * u32::try_from(size_of::<Complex32>())?,
        };

        unsafe {
            ifft_col_builder.launch(cfg)?;
        }

        Ok(())
    }

    fn gpu_conv_norm_pass<T: PrecisionType>(
        &self,
        cur_layer: &ConvBlock<T>,
        input: &Tensor4D<T>,
        batch_size: usize,
        allow_pass_2: bool,
    ) -> Result<(), Error> {
        let f_cfg = cur_layer.get_filter_cfg();
        let norm = cur_layer.get_normalisation();
        let norm_mode_u32 = u32::try_from(norm.ordinal())?;

        let norm_cache = cur_layer.get_norm_cache().unwrap();
        let norm_output = if allow_pass_2 {
            cur_layer.get_preact_features()
        } else {
            cur_layer.get_features()
        };

        let o_size = f_cfg.elements_from_length(&(input.width(), input.height()));
        let ow_u32 = u32::try_from(o_size.0)?;
        let oh_u32 = u32::try_from(o_size.1)?;
        let oc_u32 = u32::try_from(norm_output.channels())?;

        let mut builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.conv_forward_pass_func1.0,
            Precision::FP16 => &self.conv_forward_pass_func1.1,
        });

        builder
            .arg(norm_output.get_data())
            .arg(norm_cache.centered_features.get_data())
            .arg(norm_cache.prenorm_features.get_data())
            .arg(norm_cache.norm_weights.get_data())
            .arg(norm_cache.norm_biases.get_data())
            .arg(norm_cache.norm_rstd.get_data())
            .arg(&ow_u32)
            .arg(&oh_u32)
            .arg(&oc_u32)
            .arg(&batch_size)
            .arg(&norm_mode_u32);

        let grid_x = if *norm == Normalisation::BatchNorm {
            oc_u32 as usize
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

        Ok(())
    }

    fn gpu_conv_activation_pool_dropout_pass<T: PrecisionType>(
        &self,
        cur_layer: &ConvBlock<T>,
        input: &Tensor4D<T>,
        use_dropout: bool,
        step: usize,
    ) -> Result<(), Error> {
        let f_cfg = cur_layer.get_filter_cfg();
        let p_cfg = cur_layer.get_pooling_cfg();
        let p_stride = p_cfg.get_stride();
        let p_dilation = p_cfg.get_dilation();
        let act = cur_layer.get_activation();
        let pool_mode = cur_layer.get_pooling_type();
        let mask_coeff = cur_layer.get_mask_coeff();
        let seed = scramble_seed(
            u32::try_from(step)?,
            u32::try_from(cur_layer.get_filter_weights().get_id())?,
        );

        let n_u32 = u32::try_from(input.batches())?;
        let act_mode_u32 = u32::try_from(act.ordinal())?;
        let pool_mode_u32 = u32::try_from(pool_mode.ordinal())?;
        let use_dropout_u32 = u32::from(use_dropout);
        let leaky_relu_coeff = match act {
            Activation::LeakyReLU { coeff: alpha } => *alpha,
            _ => 0.0,
        };

        let features = cur_layer.get_features();

        // the 'input' here is the result of the filter/normalisation
        let f_o_size = f_cfg.elements_from_length(&(input.width(), input.height()));
        let iw_u32 = u32::try_from(f_o_size.0)?;
        let ih_u32 = u32::try_from(f_o_size.1)?;

        // the `output` here is the result of pooling
        let o_size = p_cfg.elements_from_length(&(iw_u32 as usize, ih_u32 as usize));
        let ow_u32 = u32::try_from(o_size.0)?;
        let oh_u32 = u32::try_from(o_size.1)?;
        let oc_u32 = u32::try_from(features.channels())?;

        let pw_u32 = u32::try_from(p_cfg.get_dimension().0)?;
        let ph_u32 = u32::try_from(p_cfg.get_dimension().1)?;
        let p_pad_u32 = u32::try_from(p_cfg.get_pad())?;
        let p_pad_mode_u32 = u32::try_from(p_cfg.get_pad_type().ordinal())?;
        let p_stride_x_u32 = u32::try_from(p_stride.0)?;
        let p_stride_y_u32 = u32::try_from(p_stride.1)?;
        let p_dilation_x_u32 = u32::try_from(p_dilation.0)?;
        let p_dilation_y_u32 = u32::try_from(p_dilation.1)?;

        let mut builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.conv_forward_pass_func2.0,
            Precision::FP16 => &self.conv_forward_pass_func2.1,
        });

        builder
            .arg(features.get_data())
            .arg(cur_layer.get_predrop_features().get_data())
            .arg(cur_layer.get_prepooling_features().get_data())
            .arg(cur_layer.get_preact_features().get_data())
            .arg(cur_layer.get_mask().get_data())
            .arg(&use_dropout_u32)
            .arg(&pool_mode_u32)
            .arg(&oc_u32)
            .arg(&iw_u32)
            .arg(&ih_u32)
            .arg(&ow_u32)
            .arg(&oh_u32)
            .arg(&pw_u32)
            .arg(&ph_u32)
            .arg(&p_pad_u32)
            .arg(&p_pad_mode_u32)
            .arg(&p_stride_x_u32)
            .arg(&p_stride_y_u32)
            .arg(&p_dilation_x_u32)
            .arg(&p_dilation_y_u32)
            .arg(&act_mode_u32)
            .arg(&leaky_relu_coeff)
            .arg(&mask_coeff)
            .arg(&seed);

        let tile_h = (self.tile_dim - 1) * u32::try_from(p_stride.1)?
            + u32::try_from(p_cfg.actual_height())?;
        let mut tile_w =
            (self.tile_dim - 1) * u32::try_from(p_stride.0)? + u32::try_from(p_cfg.actual_width())?;

        if tile_w.is_multiple_of(32) {
            tile_w += 1;
        }

        let cfg = self.calculate_cfg4d(
            n_u32 as usize,
            oc_u32 as usize,
            oh_u32 as usize,
            ow_u32 as usize,
            tile_w * tile_h * u32::try_from(size_of::<T>())?,
        )?;

        unsafe {
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    pub(crate) fn gpu_conv_compute_output_layer_error<T: PrecisionType>(
        &self,
        cur_layer: &ConvBlock<T>,
        target: &Tensor4D<T>,
        err_mode: LossFunc,
        activation: Activation,
    ) -> Result<(), Error> {
        let features = cur_layer.get_features();
        let m = features.batches();

        let m_u32 = u32::try_from(m)?;
        let err_mode_u32 = err_mode as u32;
        let norm_mode_u32 = u32::try_from(cur_layer.get_normalisation().ordinal())?;
        let act_mode_u32 = u32::try_from(activation.ordinal())?;

        let leaky_relu_coeff = match activation {
            Activation::LeakyReLU { coeff } => coeff,
            _ => 0.0,
        };

        // Run softmax if needed first
        if activation == Activation::Softmax {
            let cols_u32 =
                u32::try_from(features.channels() * features.width() * features.height())?;
            let mut softmax_builder = self.stream.launch_builder(match T::precision() {
                Precision::FP32 => &self.softmax_func.0,
                Precision::FP16 => &self.softmax_func.1,
            });

            softmax_builder
                .arg(features.get_data())
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

        Ok(())
    }
}
