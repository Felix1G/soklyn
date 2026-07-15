use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::io::device::GpuContext;
use crate::{Activation, ConvBlock, Normalisation, Precision, PrecisionType};
use crate::core::{scramble_seed, Tensor4D};
use crate::log::Error;

impl GpuContext {
    pub(crate) fn gpu_conv_forward_pass<T: PrecisionType>(
        &self, cur_layer: &ConvBlock<T>, input: &Tensor4D<T>, batch_size: usize,
        use_dropout: bool, step: usize
    ) -> Result<(), Error> {
        let f_cfg = cur_layer.get_filter_cfg();
        let f_stride = f_cfg.get_stride();
        let f_dilation = f_cfg.get_dilation();
        let norm = cur_layer.get_normalisation();
        let act = cur_layer.get_activation();
        let mask_coeff = cur_layer.get_mask_coeff();
        let seed = scramble_seed(step as u32, cur_layer.get_filter_weights().get_id() as u32);
        let use_bias = *norm != Normalisation::BatchNorm;

        let output = if *norm == Normalisation::Disabled {
            cur_layer.get_preact_features()
        } else {
            &cur_layer.get_norm_cache().unwrap().prenorm_features
        };
        let filter_weights = cur_layer.get_filter_weights();

        let leaky_relu_coeff = match act {
            Activation::LeakyReLU(value) => *value,
            _ => 0.0,
        };

        let use_bias_u32 = use_bias as u32;
        let pad_mode_u32 = f_cfg.get_pad_type().ordinal() as u32;
        let n_u32 = input.batches() as u32;
        let ic_u32 = input.channels() as u32;
        let oc_u32 = output.channels() as u32;
        let iw_u32 = input.width() as u32;
        let ih_u32 = input.height() as u32;
        let ow_u32 = output.width() as u32;
        let oh_u32 = output.height() as u32;
        let fw_u32 = filter_weights.width() as u32;
        let fh_u32 = filter_weights.height() as u32;
        let f_pad_u32 = f_cfg.get_pad() as u32;
        let f_stride_x_u32 = f_stride.0 as u32;
        let f_stride_y_u32 = f_stride.1 as u32;
        let f_dilation_x_u32 = f_dilation.0 as u32;
        let f_dilation_y_u32 = f_dilation.1 as u32;
        let norm_mode_u32 = norm.ordinal() as u32;
        let act_mode_u32 = act.ordinal() as u32;
        let use_dropout_u32 = use_dropout as u32;

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
            .arg(&pad_mode_u32)
            .arg(&ic_u32)
            .arg(&oc_u32)
            .arg(&iw_u32)
            .arg(&ih_u32)
            .arg(&ow_u32)
            .arg(&oh_u32)
            .arg(&fw_u32)
            .arg(&fh_u32)
            .arg(&f_pad_u32)
            .arg(&f_stride_x_u32)
            .arg(&f_stride_y_u32)
            .arg(&f_dilation_x_u32)
            .arg(&f_dilation_y_u32);

        let tile_h = (self.tile_dim - 1) * f_stride.1 as u32 + f_cfg.actual_height() as u32;
        let tile_w = (self.tile_dim - 1) * f_stride.0 as u32 + f_cfg.actual_width() as u32;

        let cfg = self.calculate_cfg4d(
            n_u32 as usize,
            oc_u32 as usize,
            oh_u32 as usize,
            ow_u32 as usize,
            tile_w * tile_h * size_of::<T>() as u32
        );

        unsafe {
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        if *norm != Normalisation::Disabled {
            let norm_cache = cur_layer.get_norm_cache().unwrap();

            let mut builder = self.stream.launch_builder(match T::precision() {
                Precision::FP32 => &self.conv_forward_pass_func1.0,
                Precision::FP16 => &self.conv_forward_pass_func1.1,
            });

            builder
                .arg(cur_layer.get_preact_features().get_data())
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
                0 //TODO
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

        Ok(())
    }
}