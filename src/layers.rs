use crate::io::device::GpuContext;
use crate::io::safetensor::{SafetensorDescriptor, read_safe_tensor, save_safe_tensor};
use crate::util::core::{Matrix, Tensor};
use crate::util::functions::Activation::Identity;
use crate::util::functions::InitFunc;
use crate::util::functions::Normalisation::Disabled;
use crate::util::functions::{
    Activation, LossFunc, Normalisation, Optimiser, Regularisation,
};
use crate::util::log::Error;
use crate::util::precision::{CastPrecision, Precision, PrecisionType};
use derivative::Derivative;
use half::f16;
use rand::rngs::ThreadRng;
use std::collections::HashMap;
use std::path::Path;

macro_rules! getter {
    ($name:ident, $($field:tt).+, $t:ty) => {
        pub fn $name(&self) -> &$t { &self.$($field).+ }
    };
}

macro_rules! getter_unwrap {
    ($name:ident, $($field:tt).+, $t:ty) => {
        pub fn $name(&self) -> &$t { &self.$($field).+.as_ref().unwrap() }
    };
}

macro_rules! getter_copy {
    ($name:ident, $($field:tt).+, $t:ty) => {
        pub fn $name(&self) -> $t { self.$($field).+ }
    };
}

macro_rules! setter {
    ($name:ident, $($field:tt).+, $t:ty) => {
        pub fn $name(&mut self, val: $t) { self.$($field).+ = val; }
    };
}

#[derive(Debug)]
struct ParamState<T: PrecisionType> {
    w: Tensor<T>,
    b: Tensor<T>,
    master_w: Option<Tensor<f32>>,
    master_b: Option<Tensor<f32>>,
    dv_w: Option<Tensor<f32>>,
    dv_b: Option<Tensor<f32>>,
    dm_w: Option<Tensor<f32>>,
    dm_b: Option<Tensor<f32>>,
}

impl<T: PrecisionType> ParamState<T> {
    fn maybe_master_tensor(
        context: &GpuContext,
        w_vec: &[T],
        shape: &[usize; 2],
    ) -> Option<Tensor<f32>> {
        match T::precision() {
            Precision::FP32 => None,
            Precision::FP16 => {
                let f32_vec = w_vec.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
                Some(Tensor::<f32>::from_cpu_vector(context, &f32_vec, shape))
            }
        }
    }

    pub fn new_linear(
        context: &GpuContext,
        w_vec: &[T],
        w_shape: &[usize; 2],
        b_shape: &[usize; 2],
        is_training: bool,
    ) -> Self {
        let get_delta_tensor = |shape: &[usize; 2]| {
            if is_training {
                Some(Tensor::<f32>::zeros(context, shape))
            } else {
                None
            }
        };

        Self {
            w: Tensor::<T>::from_cpu_vector(context, w_vec, w_shape),
            b: Tensor::<T>::zeros(context, b_shape),
            master_w: Self::maybe_master_tensor(context, w_vec, w_shape),
            master_b: match T::precision() {
                Precision::FP32 => None,
                Precision::FP16 => Some(Tensor::<f32>::zeros(context, b_shape)),
            },
            dv_w: get_delta_tensor(w_shape),
            dv_b: get_delta_tensor(b_shape),
            dm_w: get_delta_tensor(w_shape),
            dm_b: get_delta_tensor(b_shape),
        }
    }

    pub fn new_norm(context: &GpuContext, shape: &[usize; 2], is_training: bool) -> Self {
        let get_delta_tensor = || {
            if is_training {
                Some(Tensor::<f32>::zeros(context, shape))
            } else {
                None
            }
        };

        Self {
            w: Tensor::<T>::fill(context, shape, T::from_f32(1.0)),
            b: Tensor::<T>::zeros(context, shape),
            master_w: match T::precision() {
                Precision::FP32 => None,
                Precision::FP16 => Some(Tensor::<f32>::fill(context, shape, 1.0)),
            },
            master_b: match T::precision() {
                Precision::FP32 => None,
                Precision::FP16 => Some(Tensor::<f32>::zeros(context, shape)),
            },
            dv_w: get_delta_tensor(),
            dv_b: get_delta_tensor(),
            dm_w: get_delta_tensor(),
            dm_b: get_delta_tensor(),
        }
    }

    pub fn validate(&self) -> Result<(), Error> {
        if let Some(ref dv_w) = self.dv_w {
            if self.w.rows() != dv_w.rows() {
                return Err(Error::MismatchedDimensions {
                    context: "optimizer state weight vs dv_w rows",
                    expected: self.w.rows(),
                    found: dv_w.rows(),
                });
            }
            if self.w.cols() != dv_w.cols() {
                return Err(Error::MismatchedDimensions {
                    context: "optimizer state weight vs dv_w columns",
                    expected: self.w.cols(),
                    found: dv_w.cols(),
                });
            }
        }

        if let Some(ref dm_w) = self.dm_w {
            if self.w.rows() != dm_w.rows() {
                return Err(Error::MismatchedDimensions {
                    context: "optimizer state weight vs dm_w rows",
                    expected: self.w.rows(),
                    found: dm_w.rows(),
                });
            }
            if self.w.cols() != dm_w.cols() {
                return Err(Error::MismatchedDimensions {
                    context: "optimizer state weight vs dm_w columns",
                    expected: self.w.cols(),
                    found: dm_w.cols(),
                });
            }
        }

        if let Some(ref dv_b) = self.dv_b {
            if self.b.rows() != dv_b.rows() {
                return Err(Error::MismatchedDimensions {
                    context: "optimizer state bias vs dv_b rows",
                    expected: self.b.rows(),
                    found: dv_b.rows(),
                });
            }
            if self.b.cols() != dv_b.cols() {
                return Err(Error::MismatchedDimensions {
                    context: "optimizer state bias vs dv_b columns",
                    expected: self.b.cols(),
                    found: dv_b.cols(),
                });
            }
        }

        if let Some(ref dm_b) = self.dm_b {
            if self.b.rows() != dm_b.rows() {
                return Err(Error::MismatchedDimensions {
                    context: "optimizer state bias vs dm_b rows",
                    expected: self.b.rows(),
                    found: dm_b.rows(),
                });
            }
            if self.b.cols() != dm_b.cols() {
                return Err(Error::MismatchedDimensions {
                    context: "optimizer state bias vs dm_b columns",
                    expected: self.b.cols(),
                    found: dm_b.cols(),
                });
            }
        }

        Ok(())
    }
}

impl ParamState<f16> {
    fn cast_f32(self, context: &GpuContext) -> ParamState<f32> {
        let master_w = self.master_w.unwrap().clone(context);
        let master_b = self.master_b.unwrap().clone(context);

        ParamState::<f32> {
            w: master_w,
            b: master_b,
            master_w: None,
            master_b: None,
            dv_w: self.dv_w,
            dv_b: self.dv_b,
            dm_w: self.dm_w,
            dm_b: self.dm_b,
        }
    }
}

impl ParamState<f32> {
    fn cast_f16(self, context: &GpuContext) -> Result<ParamState<f16>, Error> {
        let w = self.w.clone(context);
        let b = self.b.clone(context);

        Ok(ParamState::<f16> {
            w: w.cast(context)?,
            b: b.cast(context)?,
            master_w: Some(self.w.clone(context)),
            master_b: Some(self.b.clone(context)),
            dv_w: self.dv_w,
            dv_b: self.dv_b,
            dm_w: self.dm_w,
            dm_b: self.dm_b,
        })
    }
}

#[derive(Debug)]
struct ForwardCache<T: PrecisionType> {
    out: Tensor<T>,
    predrop_out: Tensor<T>,
    preact_out: Tensor<T>,
    centered_out: Tensor<T>,
    prenorm_out: Tensor<T>,
    norm_rstd: Tensor<T>,
}

impl<T: PrecisionType> ForwardCache<T> {
    fn new(context: &GpuContext, max_batch_size: usize, outputs: usize) -> Self {
        Self {
            out: Tensor::<T>::zeros(context, &[max_batch_size, outputs]),
            predrop_out: Tensor::<T>::zeros(context, &[max_batch_size, outputs]),
            preact_out: Tensor::<T>::zeros(context, &[max_batch_size, outputs]),
            centered_out: Tensor::<T>::zeros(context, &[max_batch_size, outputs]),
            prenorm_out: Tensor::<T>::zeros(context, &[max_batch_size, outputs]),
            norm_rstd: Tensor::<T>::zeros(context, &[max_batch_size, outputs]),
        }
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

/// # The Backbone of the Entire Neural Network
/// This dense block stores all the tensors and parameters required for calculation between 2 neural network layers.
/// The `max_batch_size` is required to allocate the required cache to input and output tensors.
/// Preferably, `max_batch_size` should be the batch size used during training.
///
/// To signal that this Linear is linked to the output layer, set its [`Activation`] to [`Identity`].
///
/// The calculation goes this way:
///
/// `Input` -> `Weights` -> `Biases` -> `Normalisation` -> `Activation` -> `Mask (dropout)` -> `Output`
#[derive(Derivative)]
#[derivative(Debug)]
#[allow(dead_code)]
pub struct DenseBlock<T: PrecisionType> {
    linear_state: ParamState<T>,
    norm_state: ParamState<T>,
    forward_cache: ForwardCache<T>,
    grad: Option<Tensor<f32>>,
    d_prenorm_out: Option<Tensor<f32>>,
    d_norm_w: Option<Tensor<f32>>,
    d_norm_b: Option<Tensor<f32>>,
    mask: Tensor<T>, // for dropout
    normalisation: Normalisation,
    activation: Activation,
    regularisation: Regularisation,
    max_batch_size: usize,
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
    /// will not be generated to save memory.
    /// * `precision` - The precision of stored tensors. See [`Precision`].
    /// * `inputs` - Number of inputs.
    /// * `outputs` - Number of outputs.
    /// * `batch_size` - Maximum size of a batch. Forward and backward pass inputs/targets cannot have more batches than this.
    /// * `init` - See [`InitFunc`]. Used for initialising weights.
    pub fn default<I: InitFunc>(
        context: &GpuContext,
        is_training: bool,
        inputs: usize,
        outputs: usize,
        max_batch_size: usize,
        init: &mut I,
    ) -> Self {
        Self::new(
            context,
            is_training,
            inputs,
            outputs,
            max_batch_size,
            init,
            Disabled,
            Identity,
            Regularisation::None,
            0.0,
        )
    }

    /// Create a new [`DenseBlock`].
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `is_training` - If set to false, tensors related only to the backward pass
    /// will not be generated to save memory.
    /// * `precision` - The precision of stored tensors. See [`Precision`].
    /// * `inputs` - Number of inputs.
    /// * `outputs` - Number of outputs.
    /// * `batch_size` - Maximum size of a batch. Forward and backward pass inputs/targets cannot have more batches than this.
    /// * `init` - See [`InitFunc`]. Used for initialising weights.
    /// * `normalisation` - See [`Normalisation`]. Pass [`Disabled`] to disable normalisation.
    /// * `activation` - See [`Activation`]. Pass [`Identity`] to set this layer to an output layer.
    /// * `mask_coeff` - Mask coefficient
    pub fn new<I: InitFunc>(
        context: &GpuContext,
        is_training: bool,
        inputs: usize,
        outputs: usize,
        max_batch_size: usize,
        init: &mut I,
        normalisation: Normalisation,
        activation: Activation,
        regularisation: Regularisation,
        mask_coeff: f32,
    ) -> Self {
        let w_vec = init.init(inputs, outputs);

        let weight_shape = [inputs, outputs];
        let bias_shape = [1, outputs];

        Self::from_tensors(
            context,
            is_training,
            ParamState::new_linear(context, &w_vec, &weight_shape, &bias_shape, is_training),
            ParamState::new_norm(context, &bias_shape, is_training),
            normalisation,
            activation,
            regularisation,
            max_batch_size,
            mask_coeff,
        ).unwrap()
    }

    fn from_tensors(
        context: &GpuContext,
        is_training: bool,
        linear_state: ParamState<T>,
        norm_state: ParamState<T>,
        normalisation: Normalisation,
        activation: Activation,
        regularisation: Regularisation,
        max_batch_size: usize,
        mask_coeff: f32,
    ) -> Result<Self, Error> {
        linear_state.validate()?;
        norm_state.validate()?;

        let wc = linear_state.w.cols();
        let batch_shape = [max_batch_size, wc];
        let get_delta_tensor = || {
            if is_training {
                Some(Tensor::<f32>::zeros(context, &batch_shape))
            } else {
                None
            }
        };

        Ok(Self {
            linear_state,
            norm_state,
            forward_cache: ForwardCache::new(context, max_batch_size, wc),

            grad: get_delta_tensor(),
            d_prenorm_out: get_delta_tensor(),
            d_norm_w: get_delta_tensor(),
            d_norm_b: get_delta_tensor(),
            mask: Tensor::fill(context, &batch_shape, T::from_f32(1.0)),

            normalisation,
            activation,
            regularisation,
            max_batch_size,
            mask_coeff,
            is_training,
            rng: rand::rng(),
        })
    }

    // --- Linear State Getters ---
    getter!(get_weights, linear_state.w, Tensor<T>);
    getter!(get_biases, linear_state.b, Tensor<T>);
    getter!(
        get_master_weights,
        linear_state.master_w,
        Option<Tensor<f32>>
    );
    getter!(
        get_master_biases,
        linear_state.master_b,
        Option<Tensor<f32>>
    );
    getter_unwrap!(get_dv_weights, linear_state.dv_w, Tensor<f32>);
    getter_unwrap!(get_dv_biases, linear_state.dv_b, Tensor<f32>);
    getter_unwrap!(get_dm_weights, linear_state.dm_w, Tensor<f32>);
    getter_unwrap!(get_dm_biases, linear_state.dm_b, Tensor<f32>);

    // --- Norm State Getters ---
    getter!(get_norm_weights, norm_state.w, Tensor<T>);
    getter!(get_norm_biases, norm_state.b, Tensor<T>);
    getter!(
        get_master_norm_weights,
        norm_state.master_w,
        Option<Tensor<f32>>
    );
    getter!(
        get_master_norm_biases,
        norm_state.master_b,
        Option<Tensor<f32>>
    );
    getter_unwrap!(get_norm_weights_grad, d_norm_w, Tensor<f32>);
    getter_unwrap!(get_norm_biases_grad, d_norm_b, Tensor<f32>);
    getter_unwrap!(get_dv_norm_weights, norm_state.dv_w, Tensor<f32>);
    getter_unwrap!(get_dv_norm_biases, norm_state.dv_b, Tensor<f32>);
    getter_unwrap!(get_dm_norm_weights, norm_state.dm_w, Tensor<f32>);
    getter_unwrap!(get_dm_norm_biases, norm_state.dm_b, Tensor<f32>);

    // --- Forward Cache & Grads ---
    getter!(get_outputs, forward_cache.out, Tensor<T>);
    getter!(get_predrop_outputs, forward_cache.predrop_out, Tensor<T>);
    getter!(get_preact_outputs, forward_cache.preact_out, Tensor<T>);
    getter!(get_centered_outputs, forward_cache.centered_out, Tensor<T>);
    getter!(get_prenorm_outputs, forward_cache.prenorm_out, Tensor<T>);
    getter!(get_norm_rstd, forward_cache.norm_rstd, Tensor<T>);
    getter_unwrap!(get_grads, grad, Tensor<f32>);
    getter_unwrap!(get_delta_prenorm_out, d_prenorm_out, Tensor<f32>);
    getter!(get_masks, mask, Tensor<T>);

    // --- Config & Metadata Getters/Setters ---
    getter_copy!(get_max_batch_size, max_batch_size, usize);
    getter_copy!(get_mask_coeff, mask_coeff, f32);
    setter!(set_mask_coeff, mask_coeff, f32);
    getter!(get_normalisation, normalisation, Normalisation);
    setter!(set_normalisation, normalisation, Normalisation);
    getter!(get_activation, activation, Activation);
    setter!(set_activation, activation, Activation);
    getter!(get_regularisation, regularisation, Regularisation);
    setter!(set_regularisation, regularisation, Regularisation);

    /// Sets all the elements in the `mask` tensor back to 1.0
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    pub fn reset_mask(&mut self, context: &GpuContext) {
        self.mask.broadcast(context, T::from_f32(1.0));
    }

    /// Safely remove this block from memory.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    pub fn drop(self, context: &GpuContext) -> Result<(), Error> {
        drop(self);
        context.get_stream().synchronize()?;
        Ok(())
    }

    /// Forward propagation. The matrices are row-major i.e. batch size is determined from the
    /// number of rows of `input`.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `input` - A [`Tensor<T>`] with size `(batch size, input features)` representing
    /// the current input batch.
    /// * `batch_size` - Batch size of `input`. If the input rows exceed `batch_size`, only the
    /// first `batch_size` rows will be considered.
    /// * `use_dropout` - Whether the forward pass is part of the training loop. If set to `false`,
    /// the dropout feature will be bypassed.
    ///
    /// # Returns
    /// A [`Tensor<T>`] reference to this tensor's `out`.
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
        input: &Tensor<T>,
        batch_size: usize,
        use_dropout: bool,
        step: usize,
    ) -> Result<&Tensor<T>, Error> {
        self.check_input_dimension(input, batch_size)?;

        if self.activation == Activation::Softmax {
            return Err(Error::InvalidConfiguration {
                reason: "Softmax activation can only be passed into the compute_loss function.".to_string(),
            });
        }

        if batch_size == 0 {
            return Err(Error::InvalidBatchSize { reason: "Batch size must be more than zero." });
        }

        if self.normalisation == Normalisation::BatchNorm && batch_size == 1 {
            return Err(Error::InvalidBatchSize {
                reason: "Normalisation is set to BatchNorm, but given batch size is 1. Batch size must be more than 1.",
            });
        }

        context.gpu_forward_pass(
            self,
            input,
            batch_size,
            match self.activation {
                Activation::LeakyReLU(value) => value,
                _ => 0.0,
            },
            use_dropout,
            step,
        )?;

        Ok(&self.forward_cache.out)
    }

    fn check_input_dimension(&self, input: &Tensor<T>, batch_size: usize) -> Result<(), Error> {
        if input.rows() < batch_size {
            return Err(Error::MismatchedDimensions {
                context: "forward input rows vs explicit batch size",
                expected: batch_size,
                found: input.rows(),
            });
        }

        if input.rows() > self.max_batch_size {
            return Err(Error::AllocationLimitExceeded {
                received: input.rows(),
                max: self.max_batch_size,
            });
        }

        let expected_features = self.linear_state.w.rows();
        if input.cols() != expected_features {
            return Err(Error::MismatchedDimensions {
                context: "forward input feature columns mismatch",
                expected: expected_features,
                found: input.cols(),
            });
        }

        Ok(())
    }

    /// Computes the backward propagation for network deep learning. Output values are
    /// stored inside the network. As such, only target values are needed to be passed.
    ///
    /// The algorithm used is gradient descent. The algorithm will update the parameters (i.e. weights and biases)
    /// such that the gradient of the parameters to the error approaches 0.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `input` - A [`Tensor<T>`] that contains the input to this layer during the forward pass.
    /// * `batch_size` - Size of the batch.
    /// * `optimiser` - This [`Optimiser`] is used for linear weights and biases.
    /// * `norm_optimiser` - This [`Optimiser`] is used for normalisation weights and biases.
    /// * `learn_rate` - The learning rate of the network. Ideally, it should be between `0.0` exclusive and `1.0` inclusive.
    /// * `max_grad_norm` - Clamps gradient values between -`max_grad_norm` and +`max_grad_norm`. Pass [`f32::MAX`] to turn off clamping.
    /// * `step` - The current learning step.
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
        input: &Tensor<T>,
        batch_size: usize,
        optimiser: &Optimiser,
        norm_optimiser: &Optimiser,
        learn_rate: f32,
        max_grad_norm: f32,
        step: usize,
    ) -> Result<(), Error> {
        if !self.is_training {
            return Err(Error::TrainingModeRequired);
        }

        self.check_input_dimension(input, batch_size)?;

        if self.activation == Activation::Softmax {
            return Err(Error::InvalidConfiguration {
                reason: "Softmax activation can only be passed into the compute_loss function.".to_string(),
            });
        }

        if batch_size == 0 {
            return Err(Error::InvalidBatchSize { reason: "Batch size must be more than zero." });
        }

        if self.normalisation == Normalisation::BatchNorm && batch_size == 1 {
            return Err(Error::InvalidBatchSize {
                reason: "Normalisation is set to BatchNorm, but given batch size is 1. Batch size must be more than 1.",
            });
        }

        context.gpu_backward_pass(
            self,
            optimiser,
            norm_optimiser,
            input,
            batch_size,
            learn_rate,
            max_grad_norm,
            step,
        )
    }

    /// Computes the backward propagation for network deep learning. Output values are
    /// stored inside the network. As such, only target values are needed to be passed.
    ///
    /// The algorithm used is gradient descent. The algorithm will update the parameters (i.e. weights and biases)
    /// such that the gradient of the parameters to the error approaches 0.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `next_err` - A [`Tensor<T>`] containing the error deltas of the next layer.
    /// * `next_weights` - A [`Tensor<T>`] containing the weights of the next layer.
    /// * `input` - A [`Tensor<T>`] that contains the input to this layer during the forward pass.
    /// * `batch_size` - Size of the batch.
    /// * `optimiser` - This [`Optimiser`] is used for linear weights and biases.
    /// * `norm_optimiser` - This [`Optimiser`] is used for normalisation weights and biases.
    /// * `learn_rate` - The learning rate of the network. Ideally, it should be between `0.0` exclusive and `1.0` inclusive.
    /// * `max_grad_norm` - Clamps gradient values between -`max_grad_norm` and +`max_grad_norm`. Pass [`f32::MAX`] to turn off clamping.
    /// * `step` - The current learning step.
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
        input: &Tensor<T>,
        batch_size: usize,
        optimiser: &Optimiser,
        norm_optimiser: &Optimiser,
        learn_rate: f32,
        max_grad_norm: f32,
        step: usize,
    ) -> Result<(), Error> {
        if !self.is_training {
            return Err(Error::TrainingModeRequired);
        }

        self.check_input_dimension(input, batch_size)?;

        let current_cols = self.linear_state.w.cols();
        let next_rows = next_layer.get_weights().rows();
        if next_rows != current_cols {
            return Err(Error::MismatchedDimensions {
                context: "backprop weight adjacency (next layer rows vs current cols)",
                expected: current_cols,
                found: next_rows,
            });
        }

        let next_grad_rows = next_layer.get_grads().rows();
        if next_grad_rows != input.rows() {
            return Err(Error::MismatchedDimensions {
                context: "backprop activation gradient batch rows mismatch",
                expected: input.rows(),
                found: next_grad_rows,
            });
        }

        if self.activation == Activation::Softmax {
            return Err(Error::InvalidConfiguration {
                reason: "Softmax activation can only be passed into the compute_loss function.".to_string(),
            });
        }

        if batch_size == 0 {
            return Err(Error::InvalidBatchSize { reason: "Batch size must be more than zero." });
        }

        if self.normalisation == Normalisation::BatchNorm && batch_size == 1 {
            return Err(Error::InvalidBatchSize {
                reason: "Normalisation is set to BatchNorm, but given batch size is 1. Batch size must be more than 1.",
            });
        }

        context.gpu_hidden_layer_backward_pass(
            self,
            next_layer,
            input,
            optimiser,
            norm_optimiser,
            batch_size,
            learn_rate,
            max_grad_norm,
            &self.activation,
            step,
        )
    }

    /// Computes the error delta for this output layer.
    ///
    /// Note: dropouts are ignored i.e. output layers should not have dropouts.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `target` - A [`Tensor<T>`] with size `(batch size, output features)` representing
    /// the current target batch.
    /// * `err_mode` - See [`LossFunc`] for the available error functions.
    /// * `act_mode` - See [`Activation`] for the available activation functions.
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
        target: &Tensor<T>,
        err_mode: LossFunc,
        act_mode: Activation,
    ) -> Result<(), Error> {
        if !self.is_training {
            return Err(Error::TrainingModeRequired);
        }

        if target.cols() != self.linear_state.w.cols() {
            return Err(Error::MismatchedDimensions {
                context: "loss target columns",
                expected: self.linear_state.w.cols(),
                found: target.cols(),
            });
        }

        if target.rows() > self.max_batch_size {
            return Err(Error::AllocationLimitExceeded {
                received: target.rows(),
                max: self.max_batch_size,
            });
        }

        match err_mode {
            LossFunc::MeanSquareLoss => {
                if act_mode == Activation::Softmax {
                    return Err(Error::InvalidConfiguration {
                        reason: "Mean Squared Loss does not support the Softmax activation function.".to_string(),
                    });
                }
            }
            LossFunc::CrossEntropyLoss => {
                if act_mode != Activation::Softmax {
                    return Err(Error::InvalidConfiguration {
                        reason: "Cross-Entropy Loss only supports the Softmax activation function.".to_string(),
                    });
                }
            }
            LossFunc::BinaryCrossEntropy => {
                if act_mode != Activation::Sigmoid {
                    return Err(Error::InvalidConfiguration {
                        reason: "Binary Cross-Entropy only supports the Sigmoid activation function.".to_string(),
                    });
                }
            }
        }

        context.gpu_compute_output_layer_error(self, target, err_mode, act_mode)?;

        Ok(())
    }
}

impl DenseBlock<f32> {
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
    pub fn convert_f32(self, context: &GpuContext) -> Result<DenseBlock<f32>, Error> {
        Ok(DenseBlock::<f32> {
            linear_state: self.linear_state.cast_f32(context),
            norm_state: self.norm_state.cast_f32(context),
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

/// Saves the network into a .safetensors file.
///
/// # Arguments
/// * `context` - See [`GpuContext`].
/// * `path` - Path of the save file.
/// * `blocks` - List of [`DenseBlock`], index of which indicates its position in the network.
///
/// # Errors
/// This function will return an error if:
/// * [`Error::SerializationCasting`] - The underlying tensor raw numerical buffers fail memory alignment
///   or size validation thresholds while being transmuted into binary safe arrays via `bytemuck`.
/// * [`Error::SerdeJSON`] - The provided metadata block cannot be parsed or mapped into valid JSON string parameters.
/// * [`Error::IOError`] - The system fails to create the file at the specified `path`, hits a storage capacity allocation
///   threshold, or encounters an issue flushing the stream to disk.
pub fn save_dense_blocks<P: AsRef<Path>, T: PrecisionType>(
    context: &GpuContext,
    path: P,
    blocks: &Vec<&DenseBlock<T>>
) -> Result<(), Error> {
    save_dense_blocks_internal::<P, T, &str, &str>(context, path, blocks, None)
}

/// Saves the network into a .safetensors file.
///
/// # Arguments
/// * `context` - See [`GpuContext`].
/// * `path` - Path of the save file.
/// * `blocks` - List of [`DenseBlock`], index of which indicates its position in the network.
/// * `meta` - Metadata to be included inside the save file.
///
/// # Errors
/// This function will return an error if:
/// * [`Error::SerializationCasting`] - The underlying tensor raw numerical buffers fail memory alignment
///   or size validation thresholds while being transmuted into binary safe arrays via `bytemuck`.
/// * [`Error::SerdeJSON`] - The provided metadata block cannot be parsed or mapped into valid JSON string parameters.
/// * [`Error::IOError`] - The system fails to create the file at the specified `path`, hits a storage capacity allocation
///   threshold, or encounters an issue flushing the stream to disk.
pub fn save_dense_blocks_with_metadata<P: AsRef<Path>, T: PrecisionType, K, V>(
    context: &GpuContext,
    path: P,
    blocks: &Vec<&DenseBlock<T>>,
    meta: &[(K, V)],
) -> Result<(), Error>
where
    K: AsRef<str>,
    V: ToString,
{
    save_dense_blocks_internal(context, path, blocks, Some(meta))
}

fn save_dense_blocks_internal<P: AsRef<Path>, T: PrecisionType, K: AsRef<str>, V: ToString>(
    context: &GpuContext,
    path: P,
    blocks: &[&DenseBlock<T>],
    meta: Option<&[(K, V)]>,
) -> Result<(), Error> {
    let metadata: Vec<(String, String)> = meta
        .map(|m| {
            m.iter()
                .map(|(k, v)| (k.as_ref().to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let mut descriptors = Vec::<SafetensorDescriptor>::new();

    for (idx, layer) in blocks.iter().enumerate() {
        let n = idx + 1;

        let weights = layer.get_weights();
        let biases = layer.get_biases();
        let norm_w = layer.get_norm_weights();
        let norm_b = layer.get_norm_biases();
        let w_shape = vec![weights.rows(), weights.cols()];
        let b_shape = vec![biases.cols()];
        let nw_shape = vec![norm_w.cols()];
        let nb_shape = vec![norm_b.cols()];

        // T-precision tensors
        let t_params: &[(&str, Vec<usize>, Vec<T>)] = &[
            ("w", w_shape.clone(), weights.download(context).v),
            ("b", b_shape.clone(), biases.download(context).v),
            ("norm_w", nw_shape.clone(), norm_w.download(context).v),
            ("norm_b", nb_shape.clone(), norm_b.download(context).v),
        ];
        for (suffix, shape, data) in t_params {
            let byte_data = bytemuck::try_cast_slice(data)
                .map_err(|e| Error::SerializationCasting(format!("{e:?}")))?
                .to_vec();

            descriptors.push(SafetensorDescriptor {
                name: format!("layer{n}.{suffix}"),
                shape: shape.clone(),
                data: byte_data,
                precision: T::precision(),
            });
        }

        if layer.is_training {
            // F32 tensor: moments + master weights
            let mut f32_params: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
                (
                    format!("layer{n}.dv_w"),
                    w_shape.clone(),
                    layer.get_dv_weights().download(context).v,
                ),
                (
                    format!("layer{n}.dv_b"),
                    b_shape.clone(),
                    layer.get_dv_biases().download(context).v,
                ),
                (
                    format!("layer{n}.dm_w"),
                    w_shape.clone(),
                    layer.get_dm_weights().download(context).v,
                ),
                (
                    format!("layer{n}.dm_b"),
                    b_shape.clone(),
                    layer.get_dm_biases().download(context).v,
                ),
                (
                    format!("layer{n}.dv_norm_w"),
                    nw_shape.clone(),
                    layer.get_dv_norm_weights().download(context).v,
                ),
                (
                    format!("layer{n}.dv_norm_b"),
                    nb_shape.clone(),
                    layer.get_dv_norm_biases().download(context).v,
                ),
                (
                    format!("layer{n}.dm_norm_w"),
                    nw_shape.clone(),
                    layer.get_dm_norm_weights().download(context).v,
                ),
                (
                    format!("layer{n}.dm_norm_b"),
                    nb_shape.clone(),
                    layer.get_dm_norm_biases().download(context).v,
                ),
            ];

            // Optional master weight tensors (only present for F16 networks)
            for (suffix, tensor_opt) in [
                ("master_w", layer.get_master_weights()),
                ("master_b", layer.get_master_biases()),
                ("master_norm_w", layer.get_master_norm_weights()),
                ("master_norm_b", layer.get_master_norm_biases()),
            ] {
                if let Some(tensor) = tensor_opt {
                    f32_params.push((
                        format!("layer{n}.{suffix}"),
                        vec![tensor.rows(), tensor.cols()],
                        tensor.download(context).v,
                    ));
                }
            }

            for (name, shape, data) in f32_params {
                descriptors.push(SafetensorDescriptor {
                    name,
                    shape,
                    data: bytemuck::cast_slice(&data).to_vec(),
                    precision: Precision::FP32,
                });
            }
        }
    }

    save_safe_tensor(path, metadata, descriptors)?;

    Ok(())
}

/// Loads the tensors from a .safetensor file and creates a list of [`DenseBlock`] from them.
///
/// # Arguments
/// * `context` - See [`GpuContext`]
/// * `path` - Path to the .safetensors file
/// * `is_training` - If set to false, tensors related only to the backward pass
/// will not be generated to save memory.
/// * `max_batch_size` - Maximum size of a batch
///
/// # Errors
/// This function will return an error if:
/// * [`Error::InvalidTensorName`] - A tensor identifier starts with "layer" but violates structural format rules
///   (e.g., missing dot separator, unparseable layer index, or unrecognised parameter type suffix).
/// * [`Error::PrecisionMatch`] - A parameter marked strictly for FP32 processing (such as Adam momentum arrays)
///   is found encoded as FP16.
/// * [`Error::NoLayersFound`] - The target file contains no valid layer data entries (`max_layer == 0`).
/// * [`Error::MissingLayer`] - A layer structure index within the calculated sequence range `(1..=max_layer)`
///   is entirely missing from the file.
/// * [`Error::MissingWeights`] / [`Error::MissingBiases`] - A layer block is found, but its foundational weights
///   (`w`) or biases (`b`) matrices are missing.
/// * [`Error::DriverError`] - An asynchronous hardware or allocation failure occurs while transferring the reconstructed
///   matrices into live GPU memory buffers.
/// * Any underlying disk file-reading or foundational SafeTensors parsing fault occurs (`read_safe_tensor`).
pub fn load_dense_blocks<P: AsRef<Path>, T: PrecisionType + Default>(
    context: &GpuContext,
    path: P,
    is_training: bool,
    max_batch_size: usize,
) -> Result<Vec<DenseBlock<T>>, Error> {
    #[derive(PartialEq)]
    enum ParamType {
        W,
        B,
        MasterW,
        MasterB,
        NormW,
        NormB,
        MasterNormW,
        MasterNormB,
        DvW,
        DvB,
        DmW,
        DmB,
        DvNormW,
        DvNormB,
        DmNormW,
        DmNormB,
    }

    impl ParamType {
        fn from_str(s: &str) -> Option<Self> {
            match s {
                "w" => Some(Self::W),
                "b" => Some(Self::B),
                "master_w" => Some(Self::MasterW),
                "master_b" => Some(Self::MasterB),
                "norm_w" => Some(Self::NormW),
                "norm_b" => Some(Self::NormB),
                "master_norm_w" => Some(Self::MasterNormW),
                "master_norm_b" => Some(Self::MasterNormB),
                "dv_w" => Some(Self::DvW),
                "dv_b" => Some(Self::DvB),
                "dm_w" => Some(Self::DmW),
                "dm_b" => Some(Self::DmB),
                "dv_norm_w" => Some(Self::DvNormW),
                "dv_norm_b" => Some(Self::DvNormB),
                "dm_norm_w" => Some(Self::DmNormW),
                "dm_norm_b" => Some(Self::DmNormB),
                _ => None,
            }
        }

        fn is_f32_only(&self) -> bool {
            matches!(
                self,
                Self::MasterW
                    | Self::MasterB
                    | Self::MasterNormW
                    | Self::MasterNormB
                    | Self::DvW
                    | Self::DvB
                    | Self::DmW
                    | Self::DmB
                    | Self::DvNormW
                    | Self::DvNormB
                    | Self::DmNormW
                    | Self::DmNormB
            )
        }

        fn is_vector(&self) -> bool {
            matches!(
                self,
                Self::B
                    | Self::DvB
                    | Self::DmB
                    | Self::NormW
                    | Self::NormB
                    | Self::MasterNormW
                    | Self::MasterNormB
                    | Self::DvNormW
                    | Self::DvNormB
                    | Self::DmNormW
                    | Self::DmNormB
            )
        }
    }

    #[derive(Default)]
    struct LayerParams<T: PrecisionType> {
        w: Option<Matrix<T>>,
        b: Option<Matrix<T>>,
        master_w: Option<Matrix<f32>>,
        master_b: Option<Matrix<f32>>,
        norm_w: Option<Matrix<T>>,
        norm_b: Option<Matrix<T>>,
        master_norm_w: Option<Matrix<f32>>,
        master_norm_b: Option<Matrix<f32>>,
        dv_w: Option<Matrix<f32>>,
        dv_b: Option<Matrix<f32>>,
        dm_w: Option<Matrix<f32>>,
        dm_b: Option<Matrix<f32>>,
        dv_norm_w: Option<Matrix<f32>>,
        dv_norm_b: Option<Matrix<f32>>,
        dm_norm_w: Option<Matrix<f32>>,
        dm_norm_b: Option<Matrix<f32>>,
    }

    let tensors = read_safe_tensor(path)?;
    let mut layers_map: HashMap<usize, LayerParams<T>> = HashMap::new();
    let mut max_layer = 0usize;

    for tensor in &tensors {
        let name = &tensor.name;

        // Parse "layerN.param_suffix"
        let Some(rest) = name.strip_prefix("layer") else {
            log::debug!("Skipping non-layer key: '{name}'");
            continue;
        };

        let (layer_str, param_str) = rest.split_once('.')
            .ok_or_else(|| Error::InvalidTensorName {
                name: name.clone(),
                reason: "missing '.' separator"
            })?;

        let layer_idx = layer_str.parse::<usize>()
            .map_err(|_| Error::InvalidTensorName {
                name: name.clone(),
                reason: "invalid layer index number"
            })?;

        let param_type = ParamType::from_str(param_str)
            .ok_or_else(|| Error::InvalidTensorName {
                name: name.clone(),
                reason: "unrecognized parameter type"
            })?;

        // F32-only params must not be stored as F16
        if param_type.is_f32_only() && tensor.precision == Precision::FP16 {
            return Err(Error::PrecisionMatch {
                layer: String::from(layer_str),
                param: String::from(param_str),
            });
        }

        let (rows, cols) = if param_type.is_vector() {
            (1, tensor.shape[0])
        } else {
            (tensor.shape[0], tensor.shape[1])
        };

        let entry = layers_map.entry(layer_idx).or_default();
        max_layer = max_layer.max(layer_idx);

        if param_type.is_f32_only() {
            let mut mat: Matrix<f32> = Matrix::new(rows, cols);
            mat.v = bytemuck::pod_collect_to_vec(&tensor.data);
            match param_type {
                ParamType::MasterW => entry.master_w = Some(mat),
                ParamType::MasterB => entry.master_b = Some(mat),
                ParamType::MasterNormW => entry.master_norm_w = Some(mat),
                ParamType::MasterNormB => entry.master_norm_b = Some(mat),
                _ => {
                    if is_training {
                        match param_type {
                            ParamType::DvW => entry.dv_w = Some(mat),
                            ParamType::DvB => entry.dv_b = Some(mat),
                            ParamType::DmW => entry.dm_w = Some(mat),
                            ParamType::DmB => entry.dm_b = Some(mat),
                            ParamType::DvNormW => entry.dv_norm_w = Some(mat),
                            ParamType::DvNormB => entry.dv_norm_b = Some(mat),
                            ParamType::DmNormW => entry.dm_norm_w = Some(mat),
                            ParamType::DmNormB => entry.dm_norm_b = Some(mat),
                            _ => {}
                        }
                    }
                }
            }
        } else {
            let mut mat: Matrix<T> = Matrix::new(rows, cols);
            mat.v = bytemuck::pod_collect_to_vec(&tensor.data);
            match param_type {
                ParamType::W => entry.w = Some(mat),
                ParamType::B => entry.b = Some(mat),
                ParamType::NormW => entry.norm_w = Some(mat),
                ParamType::NormB => entry.norm_b = Some(mat),
                _ => {}
            }
        }
    }

    if max_layer == 0 {
        return Err(Error::NoLayersFound);
    }

    // Helper closures
    let from_t = |opt: &Option<Matrix<T>>, rows, cols| match opt {
        Some(m) => Tensor::<T>::from_cpu_vector(context, &m.v, &[rows, cols]),
        None => Tensor::<T>::zeros(context, &[rows, cols]),
    };

    let from_f32_opt = |opt: &Option<Matrix<f32>>, rows, cols| {
        opt.as_ref()
            .map(|m| Tensor::<f32>::from_cpu_vector(context, &m.v, &[rows, cols]))
    };

    (1..=max_layer)
        .map(|idx| {
            let layer = layers_map
                .remove(&idx)
                .ok_or(Error::MissingLayer { layer: idx })?;
            let w = layer
                .w
                .as_ref()
                .ok_or(Error::MissingWeights { layer: idx })?;
            let b = layer
                .b
                .as_ref()
                .ok_or(Error::MissingBiases { layer: idx })?;

            let (wr, wc) = (w.rows, w.cols);
            let (br, bc) = (b.rows, b.cols);

            DenseBlock::<T>::from_tensors(
                context,
                is_training,
                ParamState {
                    w: Tensor::<T>::from_cpu_vector(context, &w.v, &[wr, wc]),
                    b: Tensor::<T>::from_cpu_vector(context, &b.v, &[br, bc]),
                    master_w: from_f32_opt(&layer.master_w, wr, wc),
                    master_b: from_f32_opt(&layer.master_b, br, bc),
                    dv_w: from_f32_opt(&layer.dv_w, wr, wc),
                    dv_b: from_f32_opt(&layer.dv_b, br, bc),
                    dm_w: from_f32_opt(&layer.dm_w, wr, wc),
                    dm_b: from_f32_opt(&layer.dm_b, br, bc),
                },
                ParamState {
                    w: from_t(&layer.norm_w, br, wc),
                    b: from_t(&layer.norm_b, br, bc),
                    master_w: from_f32_opt(&layer.master_norm_w, br, wc),
                    master_b: from_f32_opt(&layer.master_norm_b, br, bc),
                    dv_w: from_f32_opt(&layer.dv_norm_w, br, wc),
                    dv_b: from_f32_opt(&layer.dv_norm_b, br, bc),
                    dm_w: from_f32_opt(&layer.dm_norm_w, br, wc),
                    dm_b: from_f32_opt(&layer.dm_norm_b, br, bc),
                },
                Disabled,
                Identity,
                Regularisation::None,
                max_batch_size,
                0.0,
            )
        })
        .collect::<Result<Vec<DenseBlock<T>>, Error>>()
}
