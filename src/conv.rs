use crate::core::{Tensor1D, Tensor4D};
use crate::io::device::GpuContext;
use crate::log::Error;
use crate::r#type::CastPrecision;
use crate::Activation::Identity;
use crate::Normalisation::Disabled;
use crate::{getter, getter_copy, setter, Activation, KernelConfig, InitFunc, Normalisation, PrecisionType, Regularisation, getter_option, PoolingType};
use half::f16;
use crate::PoolingType::MaxPooling;

pub(crate) struct ConvForwardCache<T: PrecisionType> {
    features: Tensor4D<T>,
    predrop_features: Tensor4D<T>,
    prepooling_features: Tensor4D<T>,
    preact_features: Tensor4D<T>
}

impl<T: PrecisionType> ConvForwardCache<T> {
    pub fn new(context: &GpuContext, pre_pooling_shape: &[usize; 4], aft_pooling_shape: &[usize; 4], enable_pooling: bool) -> Result<Self, Error> {
        Ok(Self {
            features: Tensor4D::zeros(context, aft_pooling_shape)?,
            predrop_features: Tensor4D::zeros(context, aft_pooling_shape)?,
            prepooling_features: Tensor4D::zeros(context, if enable_pooling {
                                                    pre_pooling_shape
                                                } else {
                                                    &[0, 0, 0, 0]
                                                })?,
            preact_features: Tensor4D::zeros(context, pre_pooling_shape)?,
        })
    }

    fn cast<U: PrecisionType>(self, context: &GpuContext) -> Result<ConvForwardCache<U>, Error> {
        Ok(ConvForwardCache::<U> {
            features: self.features.cast(context)?,
            predrop_features: self.predrop_features.cast(context)?,
            prepooling_features: self.prepooling_features.cast(context)?,
            preact_features: self.preact_features.cast(context)?,
        })
    }
}

pub(crate) struct ConvNormCache<T: PrecisionType> {
    pub(crate) norm_weights: Tensor4D<T>,
    pub(crate) norm_biases: Tensor4D<T>,
    pub(crate) centered_features: Tensor4D<T>,
    pub(crate) prenorm_features: Tensor4D<T>,
    pub(crate) norm_rstd: Tensor1D<T>,
}

impl<T: PrecisionType> ConvNormCache<T> {
    pub fn new(context: &GpuContext, norm: Normalisation, feature_shape: &[usize; 4]) -> Result<Self, Error> {
        let (norm_shape, rstd_shape) =
            if norm == Normalisation::LayerNorm || norm == Normalisation::RMSNorm {
                ([1, feature_shape[1], feature_shape[2], feature_shape[3]], [feature_shape[0]])
            } else if norm == Normalisation::BatchNorm {
                ([1, feature_shape[1], 1, 1], [feature_shape[1]])
            } else {
                panic!("Should not call ConvNormCache with no norm set.");
            };

        Ok(Self {
            norm_weights: Tensor4D::fill(context, &norm_shape, T::from_f32(1.0))?,
            norm_biases: Tensor4D::zeros(context, &norm_shape)?,
            centered_features: Tensor4D::zeros(context, feature_shape)?,
            prenorm_features: Tensor4D::zeros(context, feature_shape)?,
            norm_rstd: Tensor1D::zeros(context, &rstd_shape)?,
        })
    }

    fn cast<U: PrecisionType>(self, context: &GpuContext) -> Result<ConvNormCache<U>, Error> {
        Ok(ConvNormCache::<U> {
            norm_weights: self.norm_weights.cast(context)?,
            norm_biases: self.norm_biases.cast(context)?,
            centered_features: self.centered_features.cast(context)?,
            prenorm_features: self.prenorm_features.cast(context)?,
            norm_rstd: self.norm_rstd.cast(context)?,
        })
    }
}


pub struct ConvBlock<T: PrecisionType> {
    forward_cache: ConvForwardCache<T>,
    filter_weights: Tensor4D<T>,
    filter_biases: Tensor1D<T>,
    filter_cfg: KernelConfig,
    pooling_cfg: Option<KernelConfig>,
    pooling_type: Option<PoolingType>,
    norm_cache: Option<ConvNormCache<T>>,
    mask: Tensor4D<T>,
    max_batch_size: usize,
    max_input_dim: (usize, usize),
    is_training: bool,
    normalisation: Normalisation,
    activation: Activation,
    regularisation: Regularisation,
    mask_coeff: f32,
}

impl<T: PrecisionType> ConvBlock<T> {
    /// Create a new convolutional block with default arguments, including setting
    /// activation to [`Identity`], regularisation to [`Regularisation::None`],
    /// pooling type to [`MaxPooling`], and `mask_coeff` to `0.0`.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `is_training` - If set to false, tensors related only to the backward pass will not be generated.
    /// * `max_batch_size` - Maximum size of a batch.
    /// * `max_input_dim` - Maximum width (`max_input_dim.0`) and height (`max_input_dim.1`) of the input image.
    /// * `init` - See [`InitFunc`]. Used for initialising weight filters.
    /// * `filter_cfg` - The configuration for the filters. See [`KernelConfig`]. Pass `None` to disable.
    /// * `pooling_cfg` - The configuration for pooling. See [`KernelConfig`]. Pass `None` to disable.
    /// * `in_channels` - Number of input feature channels.
    /// * `out_channels` - Number of output feature channels.
    /// * `normalisation` - See [`Normalisation`]. Pass [`Disabled`] to disable normalisation.
    pub fn default<I: InitFunc>(
        context: &GpuContext,
        is_training: bool,
        max_batch_size: usize,
        max_input_dim: &(usize, usize),
        init: &mut I,
        filter_cfg: KernelConfig,
        pooling_cfg: Option<KernelConfig>,
        in_channels: usize,
        out_channels: usize,
        normalisation: Normalisation
    ) -> Result<Self, Error> {
        Self::new(
            context,
            is_training,
            max_batch_size,
            max_input_dim,
            init,
            filter_cfg,
            pooling_cfg,
            Some(MaxPooling),
            in_channels,
            out_channels,
            Identity,
            normalisation,
            Regularisation::None,
            0.0,
        )
    }

    /// Create a new convolutional layer block.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `is_training` - If set to false, tensors related only to the backward pass will not be generated.
    /// * `max_batch_size` - Maximum size of an activation batch.
    /// * `max_input_dim` - Maximum width (`max_input_dim.0`) and height (`max_input_dim.1`) of the input image.
    /// * `init` - See [`InitFunc`]. Used for initialising weight filters.
    /// * `filter_cfg` - The configuration for the filters. See [`KernelConfig`].
    /// * `pooling_cfg` - The configuration for pooling. See [`KernelConfig`]. Pass `None` to disable.
    /// * `pooling_type` - The type of pooling. See [`PoolingType`]. Pass `None` to disable.
    /// * `in_channels` - Number of input feature channels.
    /// * `out_channels` - Number of output feature channels.
    /// * `padding` - Border pixel extension count for input boundary handling.
    /// * `activation` - See [`Activation`]. Pass [`Identity`] to set this layer to an output layer.
    /// * `normalisation` - See [`Normalisation`]. Pass [`Disabled`] to disable normalisation.
    /// * `regularisation` - See [`Regularisation`]. Pass [`Regularisation::None`] to disable regularisation.
    /// * `mask_coeff` - Mask coefficient for dropout.
    pub fn new<I: InitFunc>(
        context: &GpuContext,
        is_training: bool,
        max_batch_size: usize,
        max_input_dim: &(usize, usize),
        init: &mut I,
        filter_cfg: KernelConfig,
        pooling_cfg: Option<KernelConfig>,
        pooling_type: Option<PoolingType>,
        in_channels: usize,
        out_channels: usize,
        activation: Activation,
        normalisation: Normalisation,
        regularisation: Regularisation,
        mask_coeff: f32,
    ) -> Result<Self, Error> {
        if max_batch_size == 0 {
            return Err(Error::InvalidConfiguration {
                reason: String::from("Batch size cannot be 0."),
            });
        }

        if in_channels == 0 || out_channels == 0 {
            return Err(Error::InvalidConfiguration {
                reason: String::from("Channel size cannot be 0."),
            });
        }

        let filter_dim = filter_cfg.get_dimension();
        let fan_in = in_channels * filter_dim.0 * filter_dim.1;
        let filter_vec = init.init(
            fan_in,
            out_channels * filter_dim.0 * filter_dim.1,
            out_channels * fan_in,
        );

        let filter_out_dim = filter_cfg.elements_from_length(&max_input_dim);
        if filter_out_dim.0 == 0 || filter_out_dim.1 == 0 {
            return Err(Error::InvalidConfiguration {
                reason: String::from("The filter size is too large. The output tensor cannot be created."),
            });
        }

        let pooling_out_dim: (usize, usize);
        if let Some(cfg) = &pooling_cfg {
            pooling_out_dim = cfg.elements_from_length(&filter_out_dim);
            if pooling_out_dim.0 == 0 || pooling_out_dim.1 == 0 {
                return Err(Error::InvalidConfiguration {
                    reason: String::from("The pooling size is too large. The output tensor cannot be created."),
                });
            }
        } else {
            pooling_out_dim = filter_out_dim.clone();
        }

        let filter_out_shape = [
            max_batch_size,
            out_channels,
            filter_out_dim.1,
            filter_out_dim.0,
        ];

        Self::from_tensors(
            is_training,
            ConvForwardCache::new(
                context,
                &filter_out_shape,
                &[
                    max_batch_size,
                    out_channels,
                    pooling_out_dim.1,
                    pooling_out_dim.0,
                ],
                pooling_cfg.is_some()
            )?,
            Tensor4D::from_cpu_vector(
                context,
                &filter_vec,
                &[out_channels, in_channels, filter_dim.1, filter_dim.0],
            )?,
            Tensor1D::zeros(context, &[out_channels])?,
            filter_cfg,
            pooling_cfg,
            pooling_type,
            Tensor4D::zeros(context, &filter_out_shape)?,
            if normalisation == Disabled {
                None
            } else {
                Some(ConvNormCache::new(context, normalisation, &filter_out_shape)?)
            },
            max_batch_size,
            max_input_dim.clone(),
            activation,
            normalisation,
            regularisation,
            mask_coeff,
        )
    }

    pub(crate) fn from_tensors(
        is_training: bool,
        forward_cache: ConvForwardCache<T>,
        filter_weights: Tensor4D<T>,
        filter_biases: Tensor1D<T>,
        filter_cfg: KernelConfig,
        pooling_cfg: Option<KernelConfig>,
        pooling_type: Option<PoolingType>,
        mask: Tensor4D<T>,
        norm_cache: Option<ConvNormCache<T>>,
        max_batch_size: usize,
        max_input_dim: (usize, usize),
        activation: Activation,
        normalisation: Normalisation,
        regularisation: Regularisation,
        mask_coeff: f32,
    ) -> Result<Self, Error> {
        Ok(Self {
            forward_cache,
            filter_weights,
            filter_biases,
            filter_cfg,
            mask,
            pooling_cfg,
            pooling_type,
            norm_cache,
            max_batch_size,
            max_input_dim,
            is_training,
            activation,
            normalisation,
            regularisation,
            mask_coeff,
        })
    }

    // Getting 4D tensors
    getter!(pub get_filter_weights, filter_weights, Tensor4D<T>);
    getter!(pub get_filter_biases, filter_biases, Tensor1D<T>);
    getter!(pub get_preact_features, forward_cache.preact_features, Tensor4D<T>);
    getter!(pub get_features, forward_cache.features, Tensor4D<T>);
    getter_option!(pub(crate) get_norm_cache, norm_cache, Option<ConvNormCache<T>>);
    getter!(pub get_mask, mask, Tensor4D<T>);

    // Getting configurations
    getter!(pub get_filter_cfg, filter_cfg, KernelConfig);
    getter_option!(pub get_pooling_cfg, pooling_cfg, Option<KernelConfig>);
    getter_option!(pub get_pooling_type, pooling_type, Option<PoolingType>);
    getter_copy!(pub get_max_batch_size, max_batch_size, usize);
    getter_copy!(pub get_mask_coeff, mask_coeff, f32);
    setter!(pub set_mask_coeff, mask_coeff, f32);
    getter!(pub get_normalisation, normalisation, Normalisation);
    getter!(pub get_activation, activation, Activation);
    setter!(pub set_activation, activation, Activation);
    getter!(pub get_regularisation, regularisation, Regularisation);
    setter!(pub set_regularisation, regularisation, Regularisation);
    getter_copy!(pub is_training_mode, is_training, bool);

    /// Note: if no pooling configuration was specified, this function will have no effect.
    pub fn set_pooling_type(&mut self, pooling_type: PoolingType) {
        if self.pooling_cfg.is_some() {
            self.pooling_type = Some(pooling_type);
        }
    }

    fn check_input_dimension(&self, input: &Tensor4D<T>, batch_size: usize) -> Result<(), Error> {
        if input.batches() < batch_size {
            return Err(Error::MismatchedDimensions {
                reason: "forward input batches vs explicit batch size",
                expected: batch_size,
                found: input.batches(),
            });
        }

        if input.batches() > self.max_batch_size {
            return Err(Error::AllocationLimitExceeded {
                received: input.batches(),
                max: self.max_batch_size,
                reason: "input batches"
            });
        }

        let expected_features = self.filter_weights.channels();
        if input.channels() != expected_features {
            return Err(Error::MismatchedDimensions {
                reason: "forward input channels mismatch",
                expected: expected_features,
                found: input.channels(),
            });
        }

        let weight_width = self.filter_weights.width();
        let weight_height = self.filter_weights.height();

        if input.width() == 0 || input.width() < weight_width {
            return Err(Error::MismatchedDimensions {
                reason: "forward input width is smaller than filter width or zero",
                expected: weight_width,
                found: input.width(),
            });
        }

        if input.height() == 0 || input.height() < weight_height {
            return Err(Error::MismatchedDimensions {
                reason: "forward input height is smaller than filter height or zero",
                expected: weight_height,
                found: input.height(),
            });
        }

        if input.width() > self.max_input_dim.0 {
            return Err(Error::AllocationLimitExceeded {
                received: input.width(),
                max: self.max_input_dim.0,
                reason: "input width",
            });
        }

        if input.height() > self.max_input_dim.1 {
            return Err(Error::AllocationLimitExceeded {
                received: input.height(),
                max: self.max_input_dim.1,
                reason: "input height",
            });
        }

        if self.filter_cfg.get_pad() >= input.width() {
            return Err(Error::MismatchedDimensions {
                reason: "forward input width is larger or equal to the padding size",
                expected: self.filter_cfg.get_pad(),
                found: input.width(),
            });
        }

        if self.filter_cfg.get_pad() >= input.height() {
            return Err(Error::MismatchedDimensions {
                reason: "forward input height is larger or equal to the padding size",
                expected: self.filter_cfg.get_pad(),
                found: input.height(),
            });
        }

        Ok(())
    }

    /// Forward propagation. If the given input dimensions are smaller than the statically configured
    /// maximum input dimensions, the input tensor is padded with zeros starting from the
    /// top-left origin `(0,0)` to match the maximum boundaries before execution.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `input` - A [`Tensor4D<T>`] with size `(batch size, input features, input width, input height)` representing
    /// the current input batch.
    /// * `batch_size` - Batch size of `input`. If the input rows exceed `batch_size`, only the
    /// first `batch_size` rows will be considered.
    /// * `use_dropout` - Whether the forward pass is part of the training loop. If set to `false`,
    /// the dropout feature will be bypassed.
    /// * `step` - The current training iteration step.
    ///
    /// # Returns
    /// A [`Tensor2D<T>`] reference to this tensor's `out`.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::MismatchedDimensions`] - The raw number of `input.batches()` is less than the current execution
    ///   `batch_size`, the input channels do not match the filter's input channels, or the input spatial
    ///   dimensions are smaller than the filter's kernel dimensions.
    /// * [`Error::AllocationLimitExceeded`] - The batch size exceeds the
    ///   statically configured `max_batch_size` limit assigned during layer creation, or the input
    ///   image dimensions exceed the configured maximum input width and height boundaries.
    /// * [`Error::InvalidConfiguration`] - The layer has an invalid activation method assigned (such as `Softmax`,
    ///   which must be handled directly via the unified `compute_loss` routine).
    /// * [`Error::InvalidBatchSize`] - The execution `batch_size` is evaluated as `0`, or evaluated as `1`
    ///   while `BatchNorm` is active (which triggers an internal division-by-zero or NaN fault).
    /// * [`Error::DriverError`] - An asynchronous hardware or synchronisation failure occurs while launching or executing
    ///   the forward inference math kernels on the GPU driver.
    pub fn forward(
        &self,
        context: &GpuContext,
        input: &Tensor4D<T>,
        batch_size: usize,
        use_dropout: bool,
        step: usize,
    ) -> Result<&Tensor4D<T>, Error> {
        self.check_input_dimension(input, batch_size)?;

        if self.activation == Activation::Softmax {
            return Err(Error::InvalidConfiguration {
                reason: "Softmax activation can only be passed into the compute_loss function."
                    .to_string(),
            });
        }

        if batch_size == 0 {
            return Err(Error::InvalidBatchSize {
                reason: "Batch size must be more than zero.",
            });
        }

        if self.normalisation == Normalisation::BatchNorm && batch_size == 1 {
            return Err(Error::InvalidBatchSize {
                reason: "Normalisation is set to BatchNorm, but given batch size is 1. Batch size must be more than 1.",
            });
        }

        context.gpu_conv_forward_pass(
            &self,
            input,
            batch_size,
            use_dropout,
            step
        )?;

        Ok(&self.forward_cache.features)
    }

    fn cast<U: PrecisionType>(self, context: &GpuContext) -> Result<ConvBlock<U>, Error> {
        Ok(ConvBlock::<U> {
            filter_weights: self.filter_weights.cast(context)?,
            filter_biases: self.filter_biases.cast(context)?,
            forward_cache: self.forward_cache.cast(context)?,
            pooling_cfg: self.pooling_cfg,
            pooling_type: self.pooling_type,
            norm_cache: if let Some(cache) = self.norm_cache {
                Some(cache.cast(context)?)
            } else {
                None
            },
            mask: self.mask.cast(context)?,
            max_batch_size: self.max_batch_size,
            max_input_dim: self.max_input_dim,
            filter_cfg: self.filter_cfg,
            is_training: self.is_training,
            normalisation: self.normalisation,
            activation: self.activation,
            regularisation: self.regularisation,
            mask_coeff: self.mask_coeff,
        })
    }
}

impl ConvBlock<f32> {
    pub fn convert_f16(self, context: &GpuContext) -> Result<ConvBlock<f16>, Error> {
        self.cast::<f16>(context)
    }
}


impl ConvBlock<f16> {
    pub fn convert_f32(self, context: &GpuContext) -> Result<ConvBlock<f32>, Error> {
        self.cast::<f32>(context)
    }
}