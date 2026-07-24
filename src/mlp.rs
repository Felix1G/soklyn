use crate::{LayerInitConfig, LossConfig, TrainContext};
use crate::core::Tensor1D;
use crate::io::device::GpuContext;
use crate::util::core::Tensor2D;
use crate::util::function::InitFunc;
use crate::util::function::{Activation, LossFunc, Normalisation, Regularisation};
use crate::util::log::Error;
use crate::util::r#type::{CastPrecision, Precision, PrecisionType};
use crate::{getter, getter_copy, getter_option, getter_unwrap, setter};
use derivative::Derivative;
use half::f16;
use rand::rngs::ThreadRng;

#[derive(Debug)]
pub(crate) struct ParamState<T: PrecisionType> {
    pub(crate) w: Tensor2D<T>,
    pub(crate) b: Tensor2D<T>,
    pub(crate) master_w: Option<Tensor2D<f32>>,
    pub(crate) master_b: Option<Tensor2D<f32>>,
    pub(crate) dv_w: Option<Tensor2D<f32>>,
    pub(crate) dv_b: Option<Tensor2D<f32>>,
    pub(crate) dm_w: Option<Tensor2D<f32>>,
    pub(crate) dm_b: Option<Tensor2D<f32>>,
}

impl<T: PrecisionType> ParamState<T> {
    fn maybe_master_tensor(
        context: &GpuContext,
        w_vec: &[T],
        shape: &[usize; 2],
    ) -> Result<Option<Tensor2D<f32>>, Error> {
        match T::precision() {
            Precision::FP32 => None,
            Precision::FP16 => {
                let f32_vec = w_vec
                    .iter()
                    .map(PrecisionType::to_f32)
                    .collect::<Vec<f32>>();
                Some(Tensor2D::<f32>::from_cpu_vector(context, &f32_vec, shape))
            }
        }
        .transpose()
    }

    pub(crate) fn new_linear(
        context: &GpuContext,
        w_vec: &[T],
        w_shape: &[usize; 2],
        b_shape: &[usize; 2],
        is_training: bool,
    ) -> Result<Self, Error> {
        let get_delta_tensor = |shape: &[usize; 2]| {
            {
                if is_training {
                    Some(Tensor2D::<f32>::zeros(context, shape))
                } else {
                    None
                }
            }
            .transpose()
        };

        Ok(Self {
            w: Tensor2D::<T>::from_cpu_vector(context, w_vec, w_shape)?,
            b: Tensor2D::<T>::zeros(context, b_shape)?,
            master_w: Self::maybe_master_tensor(context, w_vec, w_shape)?,
            master_b: match T::precision() {
                Precision::FP32 => None,
                Precision::FP16 => Some(Tensor2D::<f32>::zeros(context, b_shape)?),
            },
            dv_w: get_delta_tensor(w_shape)?,
            dv_b: get_delta_tensor(b_shape)?,
            dm_w: get_delta_tensor(w_shape)?,
            dm_b: get_delta_tensor(b_shape)?,
        })
    }

    pub(crate) fn new_norm(
        context: &GpuContext,
        shape: &[usize; 2],
        is_training: bool,
    ) -> Result<Self, Error> {
        let get_delta_tensor = || {
            {
                if is_training {
                    Some(Tensor2D::<f32>::zeros(context, shape))
                } else {
                    None
                }
            }
            .transpose()
        };

        Ok(Self {
            w: Tensor2D::<T>::fill(context, shape, T::from_f32(1.0))?,
            b: Tensor2D::<T>::zeros(context, shape)?,
            master_w: match T::precision() {
                Precision::FP32 => None,
                Precision::FP16 => Some(Tensor2D::<f32>::fill(context, shape, 1.0)?),
            },
            master_b: match T::precision() {
                Precision::FP32 => None,
                Precision::FP16 => Some(Tensor2D::<f32>::zeros(context, shape)?),
            },
            dv_w: get_delta_tensor()?,
            dv_b: get_delta_tensor()?,
            dm_w: get_delta_tensor()?,
            dm_b: get_delta_tensor()?,
        })
    }

    pub(crate) fn validate(&self) -> Result<(), Error> {
        if let Some(ref dv_w) = self.dv_w {
            if self.w.rows() != dv_w.rows() {
                return Err(Error::MismatchedDimensions {
                    reason: "optimizer state weight vs dv_w rows",
                    expected: self.w.rows(),
                    found: dv_w.rows(),
                });
            }
            if self.w.cols() != dv_w.cols() {
                return Err(Error::MismatchedDimensions {
                    reason: "optimizer state weight vs dv_w columns",
                    expected: self.w.cols(),
                    found: dv_w.cols(),
                });
            }
        }

        if let Some(ref dm_w) = self.dm_w {
            if self.w.rows() != dm_w.rows() {
                return Err(Error::MismatchedDimensions {
                    reason: "optimizer state weight vs dm_w rows",
                    expected: self.w.rows(),
                    found: dm_w.rows(),
                });
            }
            if self.w.cols() != dm_w.cols() {
                return Err(Error::MismatchedDimensions {
                    reason: "optimizer state weight vs dm_w columns",
                    expected: self.w.cols(),
                    found: dm_w.cols(),
                });
            }
        }

        if let Some(ref dv_b) = self.dv_b {
            if self.b.rows() != dv_b.rows() {
                return Err(Error::MismatchedDimensions {
                    reason: "optimizer state bias vs dv_b rows",
                    expected: self.b.rows(),
                    found: dv_b.rows(),
                });
            }
            if self.b.cols() != dv_b.cols() {
                return Err(Error::MismatchedDimensions {
                    reason: "optimizer state bias vs dv_b columns",
                    expected: self.b.cols(),
                    found: dv_b.cols(),
                });
            }
        }

        if let Some(ref dm_b) = self.dm_b {
            if self.b.rows() != dm_b.rows() {
                return Err(Error::MismatchedDimensions {
                    reason: "optimizer state bias vs dm_b rows",
                    expected: self.b.rows(),
                    found: dm_b.rows(),
                });
            }
            if self.b.cols() != dm_b.cols() {
                return Err(Error::MismatchedDimensions {
                    reason: "optimizer state bias vs dm_b columns",
                    expected: self.b.cols(),
                    found: dm_b.cols(),
                });
            }
        }

        Ok(())
    }
}

impl ParamState<f16> {
    fn cast_f32(self, context: &GpuContext) -> Result<ParamState<f32>, Error> {
        let master_w = self.master_w.unwrap().clone(context)?;
        let master_b = self.master_b.unwrap().clone(context)?;

        Ok(ParamState::<f32> {
            w: master_w,
            b: master_b,
            master_w: None,
            master_b: None,
            dv_w: self.dv_w,
            dv_b: self.dv_b,
            dm_w: self.dm_w,
            dm_b: self.dm_b,
        })
    }
}

impl ParamState<f32> {
    fn cast_f16(self, context: &GpuContext) -> Result<ParamState<f16>, Error> {
        let w = self.w.clone(context)?;
        let b = self.b.clone(context)?;

        Ok(ParamState::<f16> {
            w: w.cast(context)?,
            b: b.cast(context)?,
            master_w: Some(self.w.clone(context)?),
            master_b: Some(self.b.clone(context)?),
            dv_w: self.dv_w,
            dv_b: self.dv_b,
            dm_w: self.dm_w,
            dm_b: self.dm_b,
        })
    }
}

#[derive(Debug)]
pub(crate) struct ForwardCache<T: PrecisionType> {
    out: Tensor2D<T>,
    predrop_out: Tensor2D<T>,
    preact_out: Tensor2D<T>,
    centered_out: Tensor2D<T>,
    prenorm_out: Tensor2D<T>,
    norm_rstd: Tensor1D<T>,
}

impl<T: PrecisionType> ForwardCache<T> {
    fn new(context: &GpuContext, max_batch_size: usize, outputs: usize) -> Result<Self, Error> {
        Ok(Self {
            out: Tensor2D::<T>::zeros(context, &[max_batch_size, outputs])?,
            predrop_out: Tensor2D::<T>::zeros(context, &[max_batch_size, outputs])?,
            preact_out: Tensor2D::<T>::zeros(context, &[max_batch_size, outputs])?,
            centered_out: Tensor2D::<T>::zeros(context, &[max_batch_size, outputs])?,
            prenorm_out: Tensor2D::<T>::zeros(context, &[max_batch_size, outputs])?,
            norm_rstd: Tensor1D::<T>::zeros(context, &[usize::max(max_batch_size, outputs)])?,
        })
    }

    fn cast<U: PrecisionType>(self, context: &GpuContext) -> Result<ForwardCache<U>, Error> {
        Ok(ForwardCache::<U> {
            out: self.out.cast(context)?,
            predrop_out: self.predrop_out.cast(context)?,
            preact_out: self.preact_out.cast(context)?,
            centered_out: self.centered_out.cast(context)?,
            prenorm_out: self.prenorm_out.cast(context)?,
            norm_rstd: self.norm_rstd.cast(context)?,
        })
    }
}

/// # The Backbone of the Entire Feed Forward Neural Network
/// This dense block stores all the tensors and parameters required for calculation between 2 neural network layers.
/// The `max_batch_size` is required to allocate the required cache to input and output tensors.
/// Preferably, `max_batch_size` should be the batch size used during training.
///
/// To signal that this Linear is linked to the output layer, set its [`Activation`] to [`Identity`].
///
/// The forward pass goes this way:
///
/// `Input` -> `Weights` -> `Biases` -> `Normalisation` -> `Activation` -> `Mask (dropout)` -> `Output`
#[derive(Derivative)]
#[derivative(Debug)]
#[allow(dead_code)]
pub struct DenseBlock<T: PrecisionType> {
    linear_state: ParamState<T>,
    norm_state: ParamState<T>,
    forward_cache: ForwardCache<T>,

    grad: Option<Tensor2D<f32>>,
    d_prenorm_out: Option<Tensor2D<f32>>,
    d_norm_w: Option<Tensor2D<f32>>,
    d_norm_b: Option<Tensor2D<f32>>,

    mask: Tensor2D<T>, // for dropout
    max_batch_size: usize,

    normalisation: Normalisation,
    activation: Activation,
    regularisation: Regularisation,
    mask_coeff: f32,

    is_training: bool,
    pub(crate) rng: ThreadRng,
}

impl<T: PrecisionType> DenseBlock<T> {
    /// Create a new [`DenseBlock`] with default arguments, including setting
    /// normalisation to [`Disabled`], activation to [`Identity`],
    /// regularisation to [`Regularisation::None`], and `mask_coeff` to `0.0`.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `is_training` - If set to false, tensors related only to the backward pass
    ///   will not be generated to save memory.
    /// * `precision` - The precision of stored tensors. See [`Precision`].
    /// * `inputs` - Number of inputs.
    /// * `outputs` - Number of outputs.
    /// * `batch_size` - Maximum size of a batch. Forward and backward pass inputs/targets cannot have more batches than this.
    /// * `init` - See [`InitFunc`]. Used for initialising weights.
    ///
    /// # Errors
    /// Return an [`Error`] if the given dimensions and configurations are invalid.
    pub fn new<I: InitFunc>(
        context: &GpuContext,
        is_training: bool,
        inputs: usize,
        outputs: usize,
        max_batch_size: usize,
        init: &mut I,
    ) -> Result<DenseBlock<T>, Error> {
        Self::new_with_config(
            context,
            is_training,
            inputs,
            outputs,
            max_batch_size,
            init,
            &LayerInitConfig::default(),
        )
    }

    /// Create a new [`DenseBlock`] with custom configurations.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `is_training` - If set to false, tensors related only to the backward pass
    ///   will not be generated to save memory.
    /// * `precision` - The precision of stored tensors. See [`Precision`].
    /// * `inputs` - Number of inputs.
    /// * `outputs` - Number of outputs.
    /// * `batch_size` - Maximum size of a batch. Forward and backward pass inputs/targets cannot have more batches than this.
    /// * `init` - See [`InitFunc`]. Used for initialising weights.
    /// * `init_config` - See [`LayerInitConfig`].
    ///
    /// # Errors
    /// Return an [`Error`] if the given dimensions and configurations are invalid.
    pub fn new_with_config<I: InitFunc>(
        context: &GpuContext,
        is_training: bool,
        inputs: usize,
        outputs: usize,
        max_batch_size: usize,
        init: &mut I,
        init_config: &LayerInitConfig,
    ) -> Result<Self, Error> {
        if inputs == 0 || outputs == 0 {
            return Err(Error::InvalidConfiguration {
                reason: String::from("Input or output size cannot be 0."),
            });
        }

        if max_batch_size == 0 {
            return Err(Error::InvalidConfiguration {
                reason: String::from("Batch size cannot be 0."),
            });
        }

        let w_vec = init.init(inputs, outputs, inputs * outputs);

        let weight_shape = [inputs, outputs];
        let bias_shape = [1, outputs];

        Self::from_tensors(
            context,
            is_training,
            ParamState::new_linear(context, &w_vec, &weight_shape, &bias_shape, is_training)?,
            ParamState::new_norm(context, &bias_shape, is_training)?,
            max_batch_size,
            init_config,
        )
    }

    pub(crate) fn from_tensors(
        context: &GpuContext,
        is_training: bool,
        linear_state: ParamState<T>,
        norm_state: ParamState<T>,
        max_batch_size: usize,
        init_config: &LayerInitConfig,
    ) -> Result<Self, Error> {
        linear_state.validate()?;
        norm_state.validate()?;

        let wc = linear_state.w.cols();
        let batch_shape = [max_batch_size, wc];
        let get_delta_tensor = || {
            {
                if is_training {
                    Some(Tensor2D::<f32>::zeros(context, &batch_shape))
                } else {
                    None
                }
            }
            .transpose()
        };

        Ok(Self {
            linear_state,
            norm_state,
            forward_cache: ForwardCache::new(context, max_batch_size, wc)?,

            grad: get_delta_tensor()?,
            d_prenorm_out: get_delta_tensor()?,
            d_norm_w: get_delta_tensor()?,
            d_norm_b: get_delta_tensor()?,

            mask: Tensor2D::fill(context, &batch_shape, T::from_f32(1.0))?,
            max_batch_size,

            normalisation: init_config.normalisation,
            activation: init_config.activation,
            regularisation: init_config.regularisation,
            mask_coeff: init_config.mask_coeff,
            is_training,
            rng: rand::rng(),
        })
    }

    // --- Linear State Getters ---
    getter!(pub get_weights, linear_state.w, Tensor2D<T>);
    getter!(pub get_biases, linear_state.b, Tensor2D<T>);
    getter_option!(pub get_master_weights, linear_state.master_w, Option<Tensor2D<f32>>);
    getter_option!(pub get_master_biases, linear_state.master_b, Option<Tensor2D<f32>>);
    getter_unwrap!(pub get_dv_weights, linear_state.dv_w, Tensor2D<f32>);
    getter_unwrap!(pub get_dv_biases, linear_state.dv_b, Tensor2D<f32>);
    getter_unwrap!(pub get_dm_weights, linear_state.dm_w, Tensor2D<f32>);
    getter_unwrap!(pub get_dm_biases, linear_state.dm_b, Tensor2D<f32>);

    // --- Norm State Getters ---
    getter!(pub get_norm_weights, norm_state.w, Tensor2D<T>);
    getter!(pub get_norm_biases, norm_state.b, Tensor2D<T>);
    getter_option!(pub get_master_norm_weights, norm_state.master_w, Option<Tensor2D<f32>>);
    getter_option!(pub get_master_norm_biases, norm_state.master_b, Option<Tensor2D<f32>>);
    getter_unwrap!(pub get_norm_weights_grad, d_norm_w, Tensor2D<f32>);
    getter_unwrap!(pub get_norm_biases_grad, d_norm_b, Tensor2D<f32>);
    getter_unwrap!(pub get_dv_norm_weights, norm_state.dv_w, Tensor2D<f32>);
    getter_unwrap!(pub get_dv_norm_biases, norm_state.dv_b, Tensor2D<f32>);
    getter_unwrap!(pub get_dm_norm_weights, norm_state.dm_w, Tensor2D<f32>);
    getter_unwrap!(pub get_dm_norm_biases, norm_state.dm_b, Tensor2D<f32>);

    // --- Forward Cache & Grads ---
    getter!(pub get_outputs, forward_cache.out, Tensor2D<T>);
    getter!(pub get_predrop_outputs, forward_cache.predrop_out, Tensor2D<T>);
    getter!(pub get_preact_outputs, forward_cache.preact_out, Tensor2D<T>);
    getter!(pub get_centered_outputs, forward_cache.centered_out, Tensor2D<T>);
    getter!(pub get_prenorm_outputs, forward_cache.prenorm_out, Tensor2D<T>);
    getter!(pub get_norm_rstd, forward_cache.norm_rstd, Tensor1D<T>);
    getter_unwrap!(pub get_grads, grad, Tensor2D<f32>);
    getter_unwrap!(pub get_delta_prenorm_out, d_prenorm_out, Tensor2D<f32>);
    getter!(pub get_masks, mask, Tensor2D<T>);

    // --- Config & Metadata Getters/Setters ---
    getter_copy!(pub get_max_batch_size, max_batch_size, usize);
    getter_copy!(pub get_mask_coeff, mask_coeff, f32);
    setter!(pub set_mask_coeff, mask_coeff, f32);
    getter!(pub get_normalisation, normalisation, Normalisation);
    setter!(pub set_normalisation, normalisation, Normalisation);
    getter!(pub get_activation, activation, Activation);
    setter!(pub set_activation, activation, Activation);
    getter!(pub get_regularisation, regularisation, Regularisation);
    setter!(pub set_regularisation, regularisation, Regularisation);
    getter_copy!(pub is_training_mode, is_training, bool);

    /// Sets all the elements in the `mask` tensor back to 1.0
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    ///
    /// # Errors
    /// Returns an [`Error`] if the mask fails to be broadcasted.
    pub fn reset_mask(&mut self, context: &GpuContext) -> Result<(), Error> {
        self.mask.broadcast(context, T::from_f32(1.0))
    }

    /// Safely remove this block from memory.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    ///
    /// # Errors
    /// Returns an [`Error`] if the CUDA stream is unable to synchronise.
    pub fn drop(self, context: &GpuContext) -> Result<(), Error> {
        drop(self);
        context.get_stream().synchronize()?;
        Ok(())
    }

    fn check_input_dimension(&self, input: &Tensor2D<T>, batch_size: usize) -> Result<(), Error> {
        if input.rows() < batch_size {
            return Err(Error::MismatchedDimensions {
                reason: "forward input rows vs explicit batch size",
                expected: batch_size,
                found: input.rows(),
            });
        }

        if input.rows() > self.max_batch_size {
            return Err(Error::AllocationLimitExceeded {
                received: input.rows(),
                max: self.max_batch_size,
                reason: "input batches",
            });
        }

        let expected_features = self.linear_state.w.rows();
        if input.cols() != expected_features {
            return Err(Error::MismatchedDimensions {
                reason: "forward input feature columns mismatch",
                expected: expected_features,
                found: input.cols(),
            });
        }

        Ok(())
    }

    /// Forward propagation. The matrices are row-major.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `input` - A [`Tensor2D<T>`] with size `(batch size, input features)` representing
    ///   the current input batch.
    /// * `batch_size` - Batch size of `input`. If the input rows exceed `batch_size`, only the
    ///   first `batch_size` rows will be considered.
    /// * `use_dropout` - Whether the forward pass is part of the training loop. If set to `false`,
    ///   the dropout feature will be bypassed.
    /// * `step` - The current training iteration step.
    ///
    /// # Returns
    /// A [`Tensor2D<T>`] reference to this tensor's `out`.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::MismatchedDimensions`] - The raw number of `input.rows()` is less than the current execution
    ///   `batch_size`, or the feature data columns (`input.cols()`) fail to align with the required weight rows.
    /// * [`Error::AllocationLimitExceeded`] - The runtime execution batch size (`input.rows()`) exceeds the
    ///   statically configured `max_batch_size` limit assigned during layer creation.
    /// * [`Error::InvalidConfiguration`] - The layer has an invalid activation method assigned (such as `Softmax`,
    ///   which must be handled directly via the unified `compute_loss` routine).
    /// * [`Error::InvalidBatchSize`] - The execution `batch_size` is evaluated as `0`, or evaluated as `1`
    ///   while `BatchNorm` is active (which triggers an internal division-by-zero or NaN fault).
    /// * [`Error::DriverError`] - An asynchronous hardware or synchronisation failure occurs while launching or executing
    ///   the forward inference math kernels on the GPU driver.
    pub fn forward(
        &self,
        context: &GpuContext,
        input: &Tensor2D<T>,
        batch_size: usize,
        use_dropout: bool,
        step: usize,
    ) -> Result<&Tensor2D<T>, Error> {
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

        context.gpu_forward_pass(self, input, batch_size, use_dropout, step)?;

        Ok(&self.forward_cache.out)
    }

    /// Computes the backward propagation for network deep learning. Output values are
    /// stored inside the network. As such, only target values are needed to be passed.
    ///
    /// The algorithm used is gradient descent. The algorithm will update the parameters (i.e. weights and biases)
    /// such that the gradient of the parameters to the error approaches 0.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `input` - A [`Tensor2D<T>`] that contains the input to this layer during the forward pass.
    /// * `train_ctx` - Information and hyperparameters for training.
    /// * `step` - The current training iteration step.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::TrainingModeRequired`] - The operation is called while the network is evaluated in inference mode.
    /// * [`Error::MismatchedDimensions`] - The raw number of `input.rows()` is less than the specified execution
    ///   `batch_size`, or the feature data columns (`input.cols()`) fail to align with the required weight rows.
    /// * [`Error::AllocationLimitExceeded`] - The runtime execution batch size (`input.rows()`) exceeds the
    ///   statically configured `max_batch_size` allocation limit assigned during layer creation.
    /// * [`Error::InvalidConfiguration`] - The layer has an invalid activation method assigned (such as `Softmax`,
    ///   which must be handled directly via the unified `compute_loss` routine).
    /// * [`Error::InvalidBatchSize`] - The execution `batch_size` is evaluated as `0`, or evaluated as `1`
    ///   while `BatchNorm` is active (which triggers an internal division-by-zero or NaN fault).
    /// * [`Error::DriverError`] - An asynchronous hardware or synchronisation failure occurs while launching or executing
    ///   the backpropagation math kernels on the GPU driver.
    pub fn backward_output(
        &self,
        context: &GpuContext,
        input: &Tensor2D<T>,
        train_ctx: &TrainContext,
        step: usize,
    ) -> Result<(), Error> {
        if !self.is_training {
            return Err(Error::TrainingModeRequired {
                reason: "output layer backward pass",
            });
        }

        self.check_input_dimension(input, train_ctx.batch_size)?;

        if self.activation == Activation::Softmax {
            return Err(Error::InvalidConfiguration {
                reason: "Softmax activation can only be passed into the compute_loss function."
                    .to_string(),
            });
        }

        if train_ctx.batch_size == 0 {
            return Err(Error::InvalidBatchSize {
                reason: "Batch size must be more than zero.",
            });
        }

        if self.normalisation == Normalisation::BatchNorm && train_ctx.batch_size == 1 {
            return Err(Error::InvalidBatchSize {
                reason: "Normalisation is set to BatchNorm, but given batch size is 1. Batch size must be more than 1.",
            });
        }

        context.gpu_backward_pass(self, input, train_ctx, step)
    }

    /// Computes the backward propagation for network deep learning. Output values are
    /// stored inside the network. As such, only target values are needed to be passed.
    ///
    /// The algorithm used is gradient descent. The algorithm will update the parameters (i.e. weights and biases)
    /// such that the gradient of the parameters to the error approaches 0.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `next_err` - A [`Tensor2D<T>`] containing the error deltas of the next layer.
    /// * `next_weights` - A [`Tensor2D<T>`] containing the weights of the next layer.
    /// * `input` - A [`Tensor2D<T>`] that contains the input to this layer during the forward pass.
    /// * `train_ctx` - Information and hyperparameters for training.
    /// * `step` - The current training iteration step.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::TrainingModeRequired`] - The operation is called while the network is evaluated in inference mode.
    /// * [`Error::MismatchedDimensions`] - One of the following structural layout mismatches occurs:
    ///   * The incoming `input` tensor rows are less than the specified execution `batch_size`.
    ///   * The feature columns of the `input` tensor do not align with this layer's weight rows.
    ///   * The row count of the upstream layer's weight matrix does not equal this layer's weight columns.
    ///   * The upstream gradient tensor's batch rows do not match the incoming `input` tensor rows.
    /// * [`Error::AllocationLimitExceeded`] - The runtime execution batch size (`input.rows()`) exceeds the
    ///   statically configured `max_batch_size` allocation limit.
    /// * [`Error::InvalidConfiguration`] - The hidden layer has an invalid activation method assigned
    ///   (such as `Softmax`, which is strictly reserved for the output loss layer).
    /// * [`Error::InvalidBatchSize`] - The execution `batch_size` is evaluated as `0`, or evaluated as `1`
    ///   while `BatchNorm` is active (which triggers an internal division-by-zero or NaN fault).
    /// * [`Error::DriverError`] - An asynchronous hardware or synchronisation failure occurs while launching or executing
    ///   the backpropagation math kernels on the GPU driver.
    pub fn backward_hidden(
        &self,
        context: &GpuContext,
        next_layer: &DenseBlock<T>,
        input: &Tensor2D<T>,
        train_ctx: &TrainContext,
        step: usize,
    ) -> Result<(), Error> {
        if !self.is_training {
            return Err(Error::TrainingModeRequired {
                reason: "hidden layer backward pass",
            });
        }

        self.check_input_dimension(input, train_ctx.batch_size)?;

        let current_cols = self.linear_state.w.cols();
        let next_rows = next_layer.get_weights().rows();
        if next_rows != current_cols {
            return Err(Error::MismatchedDimensions {
                reason: "backward pass weight adjacency (next layer rows vs current cols)",
                expected: current_cols,
                found: next_rows,
            });
        }

        let next_grad_rows = next_layer.get_grads().rows();
        if next_grad_rows < input.rows() {
            return Err(Error::MismatchedDimensions {
                reason: "backward pass activation gradient rows should be more than or equal to batch rows",
                expected: input.rows(),
                found: next_grad_rows,
            });
        }

        if self.activation == Activation::Softmax {
            return Err(Error::InvalidConfiguration {
                reason: "Softmax activation can only be passed into the compute_loss function."
                    .to_string(),
            });
        }

        if train_ctx.batch_size == 0 {
            return Err(Error::InvalidBatchSize {
                reason: "Batch size must be more than zero.",
            });
        }

        if self.normalisation == Normalisation::BatchNorm && train_ctx.batch_size == 1 {
            return Err(Error::InvalidBatchSize {
                reason: "Normalisation is set to BatchNorm, but given batch size is 1. Batch size must be more than 1.",
            });
        }

        context.gpu_hidden_layer_backward_pass(
            self,
            next_layer,
            input,
            self.activation,
            train_ctx,
            step,
        )
    }

    /// Computes the error delta for this output layer.
    ///
    /// Note: dropouts are ignored i.e. output layers should not have dropouts.
    /// Also, the activation will be directly applied into the `output` tensor itself.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `target` - A [`Tensor2D<T>`] with size `(batch size, output features)` representing
    ///   the current target batch.
    /// * `loss_config` - See [`LossConfig`].
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::TrainingModeRequired`] - The network is not currently in training mode.
    /// * [`Error::MismatchedDimensions`] - The number of target columns does not match the linear layer weight columns.
    /// * [`Error::AllocationLimitExceeded`] - The target batch size (rows) exceeds the configured `max_batch_size`.
    /// * [`Error::InvalidConfiguration`] - An incompatible combination of [`LossFunc`] and [`Activation`] function is provided.
    /// * Any underlying low-level CUDA driver or synchronisation failure occurs during GPU execution.
    pub fn compute_loss(
        &self,
        context: &GpuContext,
        target: &Tensor2D<T>,
        loss_config: &LossConfig,
    ) -> Result<(), Error> {
        if !self.is_training {
            return Err(Error::TrainingModeRequired {
                reason: "compute loss",
            });
        }

        if target.cols() != self.linear_state.w.cols() {
            return Err(Error::MismatchedDimensions {
                reason: "target columns",
                expected: self.linear_state.w.cols(),
                found: target.cols(),
            });
        }

        if target.rows() > self.max_batch_size {
            return Err(Error::AllocationLimitExceeded {
                received: target.rows(),
                max: self.max_batch_size,
                reason: "target batches",
            });
        }

        match loss_config.loss_func {
            LossFunc::MeanSquareLoss => {
                if loss_config.activation == Activation::Softmax {
                    return Err(Error::InvalidConfiguration {
                        reason:
                            "Mean Squared Loss does not support the Softmax activation function."
                                .to_string(),
                    });
                }
            }
            LossFunc::CrossEntropyLoss => {
                if loss_config.activation != Activation::Softmax {
                    return Err(Error::InvalidConfiguration {
                        reason: "Cross-Entropy Loss only supports the Softmax activation function."
                            .to_string(),
                    });
                }
            }
            LossFunc::BinaryCrossEntropy => {
                if loss_config.activation != Activation::Sigmoid {
                    return Err(Error::InvalidConfiguration {
                        reason:
                            "Binary Cross-Entropy only supports the Sigmoid activation function."
                                .to_string(),
                    });
                }
            }
        }

        context.gpu_compute_output_layer_error(
            self,
            target,
            loss_config.loss_func,
            loss_config.activation,
        )?;

        Ok(())
    }
}

impl DenseBlock<f32> {
    /// Converts the [`DenseBlock`] weights, states, and caches from its current
    /// numeric type to half-precision floating-point ([`f16`]).
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    ///
    /// # Errors
    /// Returns an [`Error`] if CUDA memory allocation fails on the GPU during tensor allocation,
    /// or if an underlying GPU memory copy / type conversion kernel execution fails while
    /// casting internal states and caches.
    pub fn convert_f16(self, context: &GpuContext) -> Result<DenseBlock<f16>, Error> {
        Ok(DenseBlock::<f16> {
            linear_state: self.linear_state.cast_f16(context)?,
            norm_state: self.norm_state.cast_f16(context)?,
            forward_cache: self.forward_cache.cast(context)?,

            grad: self.grad,
            d_prenorm_out: self.d_prenorm_out,
            d_norm_w: self.d_norm_w,
            d_norm_b: self.d_norm_b,
            mask: self.mask.cast(context)?,

            normalisation: self.normalisation,
            activation: self.activation,
            regularisation: self.regularisation,
            max_batch_size: self.max_batch_size,
            mask_coeff: self.mask_coeff,
            is_training: self.is_training,
            rng: self.rng,
        })
    }
}

impl DenseBlock<f16> {
    /// Converts the [`DenseBlock`] weights, states, and caches from its current
    /// numeric type to single-precision floating-point ([`f32`]).
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    ///
    /// # Errors
    /// Returns an [`Error`] if GPU memory allocation fails or if a CUDA kernel
    /// execution error occurs while casting internal caches and masks.
    pub fn convert_f32(self, context: &GpuContext) -> Result<DenseBlock<f32>, Error> {
        Ok(DenseBlock::<f32> {
            linear_state: self.linear_state.cast_f32(context)?,
            norm_state: self.norm_state.cast_f32(context)?,
            forward_cache: self.forward_cache.cast(context)?,

            grad: self.grad,
            d_prenorm_out: self.d_prenorm_out,
            d_norm_w: self.d_norm_w,
            d_norm_b: self.d_norm_b,
            mask: self.mask.cast(context)?,

            normalisation: self.normalisation,
            activation: self.activation,
            regularisation: self.regularisation,
            max_batch_size: self.max_batch_size,
            mask_coeff: self.mask_coeff,
            is_training: self.is_training,
            rng: self.rng,
        })
    }
}
