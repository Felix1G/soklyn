use std::collections::HashMap;
use std::path::Path;
use derivative::Derivative;
use rand::rngs::ThreadRng;
use crate::device::GpuContext;
use crate::util::{check_all_equal, download_cuda_slice, Matrix, Tensor};
use crate::nn::functions::{Activation, ErrorFunc, Normalisation, Optimiser, Regularisation};
use crate::nn::functions::Activation::Identity;
use crate::nn::functions::Normalisation::Disabled;
use crate::nn::InitFunc;
use crate::nn::safetensor::{read_safe_tensor, save_safe_tensor, SafetensorDescriptor};

#[derive(Debug)]
struct ParamState {
    w: Tensor,
    b: Tensor,
    dv_w: Tensor,
    dv_b: Tensor,
    dm_w: Tensor,
    dm_b: Tensor,
}

#[derive(Debug)]
struct ForwardCache {
    out: Tensor,
    predrop_out: Tensor,
    preact_out: Tensor,
    centered_out: Tensor,
    prenorm_out: Tensor,
    norm_rstd: Tensor,
}

impl ForwardCache {
    fn new(context: &GpuContext, max_batch_size: usize, outputs: usize) -> Self {
        Self {
            out: Tensor::zeros(context, &vec![max_batch_size, outputs]),
            predrop_out: Tensor::zeros(context, &vec![max_batch_size, outputs]),
            preact_out: Tensor::zeros(context, &vec![max_batch_size, outputs]),
            centered_out: Tensor::zeros(context, &vec![max_batch_size, outputs]),
            prenorm_out: Tensor::zeros(context, &vec![max_batch_size, outputs]),
            norm_rstd: Tensor::zeros(context, &vec![max_batch_size, outputs]),
        }
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
pub struct DenseBlock {
    linear_state: ParamState,
    norm_state: ParamState,
    forward_cache: ForwardCache,
    grad: Tensor,
    mask: Tensor, // for dropout
    normalisation: Normalisation,
    activation: Activation,
    regularisation: Regularisation,
    max_batch_size: usize,
    mask_coeff: f32,
    pub(crate) rng: ThreadRng
}


impl DenseBlock {
    /// Create a new [`DenseBlock`] with default arguments, including setting
    /// normalisation to [`Normalisation::Disabled`], activation to [`Activation::Identity`],
    /// regularisation to [`Regularisation::None`], and `mask_coeff` to `0.0`.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `inputs` - Number of inputs.
    /// * `outputs` - Number of outputs.
    /// * `batch_size` - Maximum size of a batch. Forward and backward pass inputs/targets cannot have more batches than this.
    /// * `init` - See [`InitFunc`]. Used for initialising weights.
    pub fn default<I: InitFunc>(context: &GpuContext, inputs: usize, outputs: usize,
                                max_batch_size: usize, init: &mut I) -> Self {
        Self::new(
            context,
            inputs,
            outputs,
            max_batch_size,
            init,
            Disabled,
            Identity,
            Regularisation::None,
            0.0
        )
    }

    /// Create a new [`DenseBlock`].
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `inputs` - Number of inputs.
    /// * `outputs` - Number of outputs.
    /// * `batch_size` - Maximum size of a batch. Forward and backward pass inputs/targets cannot have more batches than this.
    /// * `init` - See [`InitFunc`]. Used for initialising weights.
    /// * `normalisation` - See [`Normalisation`]. Pass [`Normalisation::Disabled`] to disable normalisation.
    /// * `activation` - See [`Activation`]. Pass [`Activation::Identifier`] to set this layer to an output layer.
    /// * `mask_coeff` - Mask coefficient
    pub fn new<I: InitFunc>(
        context: &GpuContext,
        inputs: usize, outputs: usize, max_batch_size: usize,
        init: &mut I, normalisation: Normalisation, activation: Activation, regularisation: Regularisation,
        mask_coeff: f32
    ) -> Self {
        let w_vec = init.init(inputs, outputs);

        Self {
            linear_state: ParamState {
                w: Tensor::from_cpu_vector(context, &w_vec, &vec![inputs, outputs]),
                b: Tensor::zeros(context, &vec![1, outputs]),
                dv_w: Tensor::zeros(context, &vec![inputs, outputs]),
                dv_b: Tensor::zeros(context, &vec![1, outputs]),
                dm_w: Tensor::zeros(context, &vec![inputs, outputs]),
                dm_b: Tensor::zeros(context, &vec![1, outputs]),
            },
            norm_state: ParamState {
                w: Tensor::fill(context, &vec![inputs, outputs], 1.0),
                b: Tensor::zeros(context, &vec![1, outputs]),
                dv_w: Tensor::zeros(context, &vec![1, outputs]),
                dv_b: Tensor::zeros(context, &vec![1, outputs]),
                dm_w: Tensor::zeros(context, &vec![1, outputs]),
                dm_b: Tensor::zeros(context, &vec![1, outputs]),
            },
            forward_cache: ForwardCache::new(context, max_batch_size, outputs),
            grad: Tensor::zeros(context, &vec![max_batch_size, outputs]),
            mask: Tensor::fill(context, &vec![max_batch_size, outputs], 1.0),
            normalisation,
            activation,
            regularisation,
            max_batch_size,
            mask_coeff,
            rng: rand::rng()
        }
    }

    fn from_tensors(
        context: &GpuContext,
        w: Tensor, b: Tensor,
        norm_w: Tensor, norm_b: Tensor,
        dv_w: Tensor, dv_b: Tensor,
        dm_w: Tensor, dm_b: Tensor,
        dv_norm_w: Tensor, dv_norm_b: Tensor,
        dm_norm_w: Tensor, dm_norm_b: Tensor,
        normalisation: Normalisation,
        activation: Activation,
        regularisation: Regularisation,
        max_batch_size: usize,
        mask_coeff: f32
    ) -> Self {
        if !check_all_equal(&vec![w.rows(), dv_w.rows(), dm_w.rows()]) {
            panic!("Weight dimension rows are unequal.");
        }

        if !check_all_equal(&vec![w.cols(), dv_w.cols(), dm_w.cols()]) {
            panic!("Weight dimension columns are unequal.");
        }

        if !check_all_equal(&vec![b.rows(), dv_b.rows(), dm_b.rows()]) {
            panic!("Bias dimension rows are unequal.");
        }

        if !check_all_equal(&vec![b.cols(), dv_b.cols(), dm_b.cols()]) {
            panic!("Bias dimension columns are unequal.");
        }

        let wc = w.cols();
        Self {
            linear_state: ParamState {
                w, b, dv_w, dv_b, dm_w, dm_b
            },
            norm_state: ParamState {
                w: norm_w,
                b: norm_b,
                dv_w: dv_norm_w,
                dv_b: dv_norm_b,
                dm_w: dm_norm_w,
                dm_b: dm_norm_b,
            },
            forward_cache: ForwardCache::new(context, max_batch_size, wc),
            grad: Tensor::zeros(context, &vec![max_batch_size, wc]),
            mask: Tensor::fill(context, &vec![max_batch_size, wc], 1.0),
            normalisation,
            activation,
            regularisation,
            max_batch_size,
            mask_coeff,
            rng: rand::rng()
        }
    }

    pub fn get_weights(&self) -> &Tensor { &self.linear_state.w }
    pub fn get_biases(&self) -> &Tensor { &self.linear_state.b }
    pub fn get_norm_weights(&self) -> &Tensor { &self.norm_state.w }
    pub fn get_norm_biases(&self) -> &Tensor { &self.norm_state.b }
    pub fn get_outputs(&self) -> &Tensor { &self.forward_cache.out }
    pub fn get_predrop_outputs(&self) -> &Tensor { &self.forward_cache.predrop_out }
    pub fn get_preact_outputs(&self) -> &Tensor { &self.forward_cache.preact_out }
    pub fn get_centered_outputs(&self) -> &Tensor { &self.forward_cache.centered_out }
    pub fn get_prenorm_outputs(&self) -> &Tensor { &self.forward_cache.prenorm_out }
    pub fn get_norm_rstd(&self) -> &Tensor { &self.forward_cache.norm_rstd }
    pub fn get_grads(&self) -> &Tensor { &self.grad }
    pub fn get_masks(&self) -> &Tensor { &self.mask }
    /// Sets all the elements in the `mask` tensor back to 1.0
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    pub fn reset_mask(&mut self, context: &GpuContext) { self.mask.broadcast(context, 1.0); }
    pub fn get_mask_coeff(&self) -> f32 { self.mask_coeff }
    pub fn set_mask_coeff(&mut self, coeff: f32) { self.mask_coeff = coeff }
    pub fn get_dv_weights(&self) -> &Tensor { &self.linear_state.dv_w }
    pub fn get_dv_biases(&self) -> &Tensor { &self.linear_state.dv_b }
    pub fn get_dm_weights(&self) -> &Tensor { &self.linear_state.dm_w }
    pub fn get_dm_biases(&self) -> &Tensor { &self.linear_state.dm_b }
    pub fn get_dv_norm_weights(&self) -> &Tensor { &self.norm_state.dv_w }
    pub fn get_dv_norm_biases(&self) -> &Tensor { &self.norm_state.dv_b }
    pub fn get_dm_norm_weights(&self) -> &Tensor { &self.norm_state.dm_w }
    pub fn get_dm_norm_biases(&self) -> &Tensor { &self.norm_state.dm_b }
    pub fn download_dv_weights(&self, context: &GpuContext) -> Vec<f32> { download_cuda_slice(context.get_stream(), &self.linear_state.dv_w.get_data()) }
    pub fn download_dv_biases(&self, context: &GpuContext) -> Vec<f32> { download_cuda_slice(context.get_stream(), &self.linear_state.dv_b.get_data()) }
    pub fn download_dm_weights(&self, context: &GpuContext) -> Vec<f32> { download_cuda_slice(context.get_stream(), &self.linear_state.dm_w.get_data()) }
    pub fn download_dm_biases(&self, context: &GpuContext) -> Vec<f32> { download_cuda_slice(context.get_stream(), &self.linear_state.dm_b.get_data()) }
    pub fn download_dv_norm_weights(&self, context: &GpuContext) -> Vec<f32> { download_cuda_slice(context.get_stream(), &self.norm_state.dv_w.get_data()) }
    pub fn download_dv_norm_biases(&self, context: &GpuContext) -> Vec<f32> { download_cuda_slice(context.get_stream(), &self.norm_state.dv_b.get_data()) }
    pub fn download_dm_norm_weights(&self, context: &GpuContext) -> Vec<f32> { download_cuda_slice(context.get_stream(), &self.norm_state.dm_w.get_data()) }
    pub fn download_dm_norm_biases(&self, context: &GpuContext) -> Vec<f32> { download_cuda_slice(context.get_stream(), &self.norm_state.dm_b.get_data()) }
    pub fn get_normalisation(&self) -> &Normalisation { &self.normalisation }
    pub fn set_normalisation(&mut self, normalisation: Normalisation) { self.normalisation = normalisation; }
    pub fn get_activation(&self) -> &Activation { &self.activation }
    pub fn set_activation(&mut self, activation: Activation) { self.activation = activation; }
    pub fn get_regularisation(&self) -> &Regularisation { &self.regularisation }
    pub fn set_regularisation(&mut self, regularisation: Regularisation) { self.regularisation = regularisation; }
    pub fn get_max_batch_size(&self) -> usize { self.max_batch_size }

    /// Forward propagation. The matrices are row-major i.e. batch size is determined from the
    /// number of rows of `input`.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `input` - A [`Tensor`] with size `(batch size, input features)` representing
    /// the current input batch.
    /// * `batch_size` - Batch size of `input`. If the input rows exceed `batch_size`, only the
    /// first `batch_size` rows will be considered.
    /// * `is_training` - Whether the forward pass is part of the training loop. If set to `false`,
    /// the dropout feature will be bypassed.
    ///
    /// # Returns
    /// A [`Tensor`] reference to this tensor's `out`.
    ///
    /// # Panics
    /// This function will panic if the `input` size does not match the expected dimensions, or
    /// the activation is set to [`Activation::Softmax`] (It can only be passed into [`DenseBlock::compute_loss`]).
    pub fn forward(&self, context: &GpuContext, input: &Tensor, batch_size: usize, is_training: bool, step: usize) -> &Tensor {
        if input.rows() < batch_size {
            panic!(
                "Input rows must be more than or equal to batch size. Expected: >= {}, Got: {}",
                batch_size, input.rows()
            );
        }

        if input.rows() > self.max_batch_size || input.cols() != self.linear_state.w.rows() {
            panic!(
                "Dimension mismatch during forward pass. Expected: (at most {})x{}, Got: {}x{}",
                self.max_batch_size, self.linear_state.w.rows(), input.rows(), input.cols()
            );
        }

        if self.activation == Activation::Softmax {
            panic!("Softmax activation can only be passed into the compute_loss function.");
        }

        assert_ne!(batch_size, 0, "Batch size must be more than zero.");
        if self.normalisation == Normalisation::BatchNorm && batch_size == 1 {
            panic!("Normalisation is set to BatchNorm, but given batch size is {}. Batch size must be more than 1.", batch_size)
        }

        context.gpu_forward_pass(
            self, input, batch_size, match self.activation {
                Activation::LeakyReLU(value) => value,
                _ => 0.0,
            }, is_training, step
        );

        &self.forward_cache.out
    }

    /// Computes the backward propagation for network deep learning. Output values are
    /// stored inside the network. As such, only target values are needed to be passed.
    ///
    /// The algorithm used is gradient descent. The algorithm will update the parameters (i.e. weights and biases)
    /// such that the gradient of the parameters to the error approaches 0.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `input` - A [`Tensor`] that contains the input to this layer during the forward pass.
    /// * `batch_size` - Size of the batch.
    /// * `optimiser` - This [`Optimiser`] is used for linear weights and biases.
    /// * `norm_optimiser` - This [`Optimiser`] is used for normalisation weights and biases.
    /// * `learn_rate` - The learning rate of the network. Ideally, it should be between `0.0` exclusive and `1.0` inclusive.
    /// * `max_grad_norm` - Clamps gradient values between -`max_grad_norm` and +`max_grad_norm`. Pass [`f32::MAX`] to turn off clamping.
    /// * `step` - The current learning step.
    ///
    /// # Panics
    /// This function will panic if the `input` size does not match the expected dimensions, or
    /// the activation is set to [`Activation::Softmax`] (It can only be passed into [`DenseBlock::compute_loss`]).
    pub fn backward_output(&self, context: &GpuContext, input: &Tensor, batch_size: usize,
                           optimiser: &Optimiser, norm_optimiser: &Optimiser,
                           learn_rate: f32, max_grad_norm: f32, step: usize) -> &Tensor {
        if input.rows() < batch_size {
            panic!(
                "Input rows must be more than or equal to batch size. Expected: >= {}, Got: {}",
                batch_size, input.rows()
            );
        }

        if input.rows() > self.max_batch_size || input.cols() != self.linear_state.w.rows() {
            panic!(
                "Input dimension mismatch during forward pass. Expected: (at most {})x{}, Got: {}x{}",
                self.max_batch_size, self.linear_state.w.rows(), input.rows(), input.cols()
            );
        }

        if self.activation == Activation::Softmax {
            panic!("Softmax activation can only be passed into the compute_loss function.");
        }

        assert_ne!(batch_size, 0, "Batch size must be more than zero.");
        if self.normalisation == Normalisation::BatchNorm && batch_size == 1 {
            panic!("Normalisation is set to BatchNorm, but given batch size is {}. Batch size must be more than 1.", batch_size)
        }
        
        context.gpu_backward_pass(self, optimiser, norm_optimiser, input, batch_size, learn_rate, max_grad_norm, step);
        &self.grad
    }

    /// Computes the backward propagation for network deep learning. Output values are
    /// stored inside the network. As such, only target values are needed to be passed.
    ///
    /// The algorithm used is gradient descent. The algorithm will update the parameters (i.e. weights and biases)
    /// such that the gradient of the parameters to the error approaches 0.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `next_err` - A [`Tensor`] containing the error deltas of the next layer.
    /// * `next_weights` - A [`Tensor`] containing the weights of the next layer.
    /// * `input` - A [`Tensor`] that contains the input to this layer during the forward pass.
    /// * `batch_size` - Size of the batch.
    /// * `optimiser` - This [`Optimiser`] is used for linear weights and biases.
    /// * `norm_optimiser` - This [`Optimiser`] is used for normalisation weights and biases.
    /// * `learn_rate` - The learning rate of the network. Ideally, it should be between `0.0` exclusive and `1.0` inclusive.
    /// * `max_grad_norm` - Clamps gradient values between -`max_grad_norm` and +`max_grad_norm`. Pass [`f32::MAX`] to turn off clamping.
    /// * `step` - The current learning step.
    ///
    /// # Panics
    /// This function will panic if the `input`, `next_err` and `next_weights` size does not match the expected dimensions, or
    /// the activation is set to [`Activation::Softmax`] (It can only be passed into [`DenseBlock::compute_loss`]).
    /// 
    /// Also panics if batch size is `0`, or if set to [`Normalisation::BatchNorm`], batch size must be more than `1`.
    pub fn backward_hidden(&self, context: &GpuContext, next_err: &Tensor, next_weights: &Tensor, input: &Tensor, batch_size: usize,
                           optimiser: &Optimiser, norm_optimiser: &Optimiser, learn_rate: f32, max_grad_norm: f32, step: usize) -> &Tensor {
        if input.rows() < batch_size {
            panic!(
                "Input rows must be more than or equal to batch size. Expected: >= {}, Got: {}",
                batch_size, input.rows()
            );
        }

        if input.rows() > self.max_batch_size || input.cols() != self.linear_state.w.rows() {
            panic!(
                "Input dimension mismatch during forward pass. Expected: (at most {})x{}, Got: {}x{}",
                self.max_batch_size, self.linear_state.w.rows(), input.rows(), input.cols()
            );
        }

        if next_weights.rows() != self.linear_state.w.cols() {
            panic!(
                "Rows of next layer's weights must equal to the columns of this layer's weights. Expected: {}, Got: {}",
                self.linear_state.w.cols(), next_weights.rows()
            );
        }

        if next_err.rows() != input.rows() {
            panic!(
                "Next layer error rows must equal to input rows. Expected: {}, Got: {}",
                input.rows(), next_err.rows()
            );
        }

        if self.activation == Activation::Softmax {
            panic!("Softmax activation can only be passed into the compute_loss function.");
        }
        
        assert_ne!(batch_size, 0, "Batch size must be more than zero.");
        if self.normalisation == Normalisation::BatchNorm && batch_size == 1 {
            panic!("Normalisation is set to BatchNorm, but given batch size is {}. Batch size must be more than 1.", batch_size)
        }

        context.gpu_hidden_layer_backpass(self, optimiser, norm_optimiser, next_err, next_weights, input,
                                          batch_size, learn_rate, max_grad_norm, &self.activation, step);
        &self.grad
    }

    /// Computes the error delta for this output layer.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `target` - A [`Tensor`] with size `(batch size, output features)` representing
    /// the current target batch.
    /// * `err_mode` - See [`ErrorFunc`] for the available error functions.
    /// * `act_mode` - See [`Activation`] for the available activation functions.
    ///
    /// # Panics
    /// This function will panic if the `target` size does not match the expected dimensions.
    pub fn compute_loss(&self, context: &GpuContext, target: &Tensor, err_mode: ErrorFunc, act_mode: Activation) -> &Tensor {
        if target.cols() != self.linear_state.w.cols() {
            panic!(
                "Dimension mismatch during loss computation. Expected: (at most {})x{}, Got: {}x{}",
                self.max_batch_size, self.linear_state.w.cols(), target.rows(), target.cols()
            );
        }

        if target.rows() > self.max_batch_size {
            panic!(
                "Target batches (received: {}) must be less than or equal to maximum batch size {}.",
                target.rows(), self.max_batch_size
            );
        }

        match err_mode {
            ErrorFunc::MeanSquareLoss => {
                assert_ne!(act_mode, Activation::Softmax, "Mean Squared Loss does not support the Softmax activation function.");
            }
            ErrorFunc::CrossEntropyLoss => {
                assert_eq!(act_mode, Activation::Softmax, "Cross-Entropy Loss only supports the Softmax activation function.");
            }
            ErrorFunc::BinaryCrossEntropy => {
                assert_eq!(act_mode, Activation::Sigmoid, "Binary Cross-Entropy only supports the Sigmoid activation function.");
            }
        }

        context.gpu_compute_output_layer_error(self, target, err_mode, act_mode);
        &self.grad
    }
}

/// Saves the network into a .safetensors file.
///
/// # Arguments
/// * `context` - See [`GpuContext`].
/// * `path` - Path of the save file.
/// * `blocks` - List of [`DenseBlock`], index of which indicates its position in the network.
pub fn save_tensors<P: AsRef<Path>>(context: &GpuContext, path: P, blocks: &Vec<&DenseBlock>) {
    save_tensors_internal::<P, &str, &str>(context, path, blocks, None);
}

/// Saves the network into a .safetensors file.
///
/// # Arguments
/// * `context` - See [`GpuContext`].
/// * `path` - Path of the save file.
/// * `blocks` - List of [`DenseBlock`], index of which indicates its position in the network.
/// * `meta` - Metadata to be included inside the save file.
pub fn save_tensors_with_metadata<P: AsRef<Path>, K, V>(context: &GpuContext, path: P, blocks: &Vec<&DenseBlock>, meta: &[(K, V)])
where
    K: AsRef<str>,
    V: ToString,
{
    save_tensors_internal(context, path, blocks, Some(meta));
}

fn save_tensors_internal<P: AsRef<Path>, K, V>(context: &GpuContext, path: P, blocks: &Vec<&DenseBlock>, meta: Option<&[(K, V)]>)
where
    K: AsRef<str>,
    V: ToString,
{
    let metadata: Vec<(String, String)> = meta
        .map(|meta_iter| {
            meta_iter
                .into_iter()
                .map(|(k, v)| (k.as_ref().to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let mut descriptor = Vec::<SafetensorDescriptor>::new();

    for (idx, layer) in blocks.iter().enumerate() {
        let layer_num = idx + 1;

        // Define a collection of all possible parameters, their naming suffixes,
        // their shape properties, data arrays, and whether they should be included.
        let weights = layer.get_weights();
        let biases = layer.get_biases();
        let norm_weights = layer.get_norm_weights();
        let norm_biases = layer.get_norm_biases();
        let params = [
            (format!("layer{layer_num}.w"), vec![weights.rows(), weights.cols()], &weights.download(&context).v),
            (format!("layer{layer_num}.b"), vec![biases.cols()], &biases.download(&context).v),
            (format!("layer{layer_num}.norm_w"), vec![norm_weights.cols()], &norm_weights.download(&context).v),
            (format!("layer{layer_num}.norm_b"), vec![norm_biases.cols()], &norm_biases.download(&context).v),
            (format!("layer{layer_num}.dv_w"), vec![weights.rows(), weights.cols()], &layer.download_dv_weights(&context)),
            (format!("layer{layer_num}.dv_b"), vec![biases.cols()], &layer.download_dv_biases(&context)),
            (format!("layer{layer_num}.dm_w"), vec![weights.rows(), weights.cols()], &layer.download_dm_weights(&context)),
            (format!("layer{layer_num}.dm_b"), vec![biases.cols()], &layer.download_dm_biases(&context)),
            (format!("layer{layer_num}.dv_norm_w"), vec![norm_weights.cols()], &layer.download_dv_norm_weights(&context)),
            (format!("layer{layer_num}.dv_norm_b"), vec![norm_biases.cols()], &layer.download_dv_norm_biases(&context)),
            (format!("layer{layer_num}.dm_norm_w"), vec![norm_weights.cols()], &layer.download_dm_norm_weights(&context)),
            (format!("layer{layer_num}.dm_norm_b"), vec![norm_biases.cols()], &layer.download_dm_norm_biases(&context)),
        ];

        // Process the parameter descriptors uniformly without repetitive if-statements
        for (name, shape, data) in params {
            descriptor.push(SafetensorDescriptor {
                name,
                shape,
                data: data.clone(),
            });
        }
    }

    save_safe_tensor(path, metadata, descriptor);
}

/// Creates a network from a .safetensor file.
///
/// # Arguments
/// * `context` - See [`GpuContext`]
/// * `path` - Path to the .safetensors file
/// * `max_batch_size` - Maximum size of a batch
pub fn load_tensors<P: AsRef<Path>>(
    context: &GpuContext, path: P, max_batch_size: usize
) -> Vec<DenseBlock> {
    enum ParamType {
        W, B, NormW, NormB, DvW, DvB, DmW, DmB, DvNormW, DvNormB, DmNormW, DmNormB,
    }

    #[derive(Default)]
    struct LayerParams {
        w: Option<Matrix>,
        b: Option<Matrix>,
        norm_w: Option<Matrix>,
        norm_b: Option<Matrix>,
        dv_w: Option<Matrix>,
        dv_b: Option<Matrix>,
        dm_w: Option<Matrix>,
        dm_b: Option<Matrix>,
        dv_norm_w: Option<Matrix>,
        dv_norm_b: Option<Matrix>,
        dm_norm_w: Option<Matrix>,
        dm_norm_b: Option<Matrix>,
    }

    let tensors = read_safe_tensor(path);
    let mut layers_map: HashMap<usize, LayerParams> = HashMap::new();

    let mut max_layers = 0;

    for tensor in &tensors {
        let name = &tensor.name;
        if !name.starts_with("layer") {
            println!("Tensor name '{name}' does not start with 'layer'! Skipping...");
            continue;
        }

        // Safely extract layer index and parameter component suffix
        let Some((layer_str, param_str)) = name["layer".len()..].split_once('.') else {
            println!("Tensor name '{name}' does not contain a '.' suffix! Skipping...");
            continue;
        };

        let Ok(layer_idx) = layer_str.parse::<usize>() else {
            println!("Tensor name '{name}' contains an invalid layer index! Skipping...");
            continue;
        };

        let param_type = match param_str {
            "w" => ParamType::W,
            "b" => ParamType::B,
            "norm_w" => ParamType::NormW,
            "norm_b" => ParamType::NormB,
            "dv_w" => ParamType::DvW,
            "dv_b" => ParamType::DvB,
            "dm_w" => ParamType::DmW,
            "dm_b" => ParamType::DmB,
            "dv_norm_w" => ParamType::DvNormW,
            "dv_norm_b" => ParamType::DvNormB,
            "dm_norm_w" => ParamType::DmNormW,
            "dm_norm_b" => ParamType::DmNormB,
            _ => {
                println!("Tensor '{name}' has unrecognized type '{param_str}'. Skipping...");
                continue;
            }
        };

        max_layers = max_layers.max(layer_idx);
        let layer_entry = layers_map.entry(layer_idx).or_default();

        // Build Matrix metadata dynamically based on type shape rules
        let (rows, cols) = match param_type {
            ParamType::B | ParamType::DvB | ParamType::DmB |
            ParamType::NormW | ParamType::NormB | ParamType::DvNormW | ParamType::DvNormB |
            ParamType::DmNormW | ParamType::DmNormB => (1, tensor.shape[0]),
            _ => (tensor.shape[0], tensor.shape[1]),
        };

        let mut mat = Matrix::new(rows, cols);
        mat.v = tensor.data.clone();

        match param_type {
            ParamType::W => layer_entry.w = Some(mat),
            ParamType::B => layer_entry.b = Some(mat),
            ParamType::NormW => layer_entry.norm_w = Some(mat),
            ParamType::NormB => layer_entry.norm_b = Some(mat),
            ParamType::DvW => layer_entry.dv_w = Some(mat),
            ParamType::DvB => layer_entry.dv_b = Some(mat),
            ParamType::DmW => layer_entry.dm_w = Some(mat),
            ParamType::DmB => layer_entry.dm_b = Some(mat),
            ParamType::DvNormW => layer_entry.dv_norm_w = Some(mat),
            ParamType::DvNormB => layer_entry.dv_norm_b = Some(mat),
            ParamType::DmNormW => layer_entry.dm_norm_w = Some(mat),
            ParamType::DmNormB => layer_entry.dm_norm_b = Some(mat),
        }
    }

    assert_ne!(max_layers, 0, "There are no network layers in this tensor file.");

    let mut blocks = Vec::<DenseBlock>::new();

    let unwrap_vec = |v: &Option<Matrix>, rows: usize, cols: usize| {
        if let Some(m) = v {
            Tensor::from_cpu_vector(&context, &m.v, &vec![rows, cols])
        } else {
            Tensor::zeros(&context, &vec![rows, cols])
        }
    };

    for idx in 1..=max_layers {
        let layer = layers_map.remove(&idx).unwrap_or_else(|| {
            panic!("Layer {idx} data is completely missing in file.");
        });

        let w_mat = layer.w.unwrap_or_else(|| panic!("Weights for layer {idx} do not exist"));
        let b_mat = layer.b.unwrap_or_else(|| panic!("Biases for layer {idx} do not exist"));

        blocks.push(DenseBlock::from_tensors(
            context,
            Tensor::from_cpu_vector(context, &w_mat.v, &vec![w_mat.rows, w_mat.cols]),
            Tensor::from_cpu_vector(context, &b_mat.v, &vec![b_mat.rows, b_mat.cols]),
            unwrap_vec(&layer.norm_w, w_mat.rows, w_mat.cols),
            unwrap_vec(&layer.norm_b, b_mat.rows, b_mat.cols),
            unwrap_vec(&layer.dv_w, w_mat.rows, w_mat.cols),
            unwrap_vec(&layer.dv_b, b_mat.rows, b_mat.cols),
            unwrap_vec(&layer.dm_w, w_mat.rows, w_mat.cols),
            unwrap_vec(&layer.dm_b, b_mat.rows, b_mat.cols),
            unwrap_vec(&layer.dv_norm_w, b_mat.rows, b_mat.cols),
            unwrap_vec(&layer.dv_norm_b, b_mat.rows, b_mat.cols),
            unwrap_vec(&layer.dm_norm_w, b_mat.rows, b_mat.cols),
            unwrap_vec(&layer.dm_norm_b, b_mat.rows, b_mat.cols),
            Disabled,
            Identity,
            Regularisation::None,
            max_batch_size,
            0.0
        ));
    }

    blocks
}