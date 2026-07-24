use crate::LossConfig;
use crate::getter;
use crate::io::device::GpuContext;
use crate::mlp::DenseBlock;
use crate::util::context::MultiLayerTrainContext;
use crate::util::core::Tensor2D;
use crate::util::log::Error;
use crate::util::r#type::PrecisionType;

pub struct FeedForwardNetwork<T: PrecisionType> {
    layers: Vec<DenseBlock<T>>,
    inputs: usize,
}

impl<T: PrecisionType> FeedForwardNetwork<T> {
    /// Creates a new `FeedForwardNetwork` instance by wrapping a sequential stack of layers.
    ///
    /// # Arguments
    /// * `layers` - A vector of `DenseBlock<T>` representing the layers of the network.
    /// * `inputs` - The total expected feature dimension length of a single raw input sample.
    #[must_use]
    pub fn new(layers: Vec<DenseBlock<T>>, inputs: usize) -> Self {
        Self { layers, inputs }
    }

    /// Executes a sequential forward execution pass.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `input` - The input tensor for the first layer.
    /// * `batch_size` - The batch size for this pass.
    /// * `use_dropout` - If set to false, dropout is disabled.
    /// * `step` - The current training iteration step.
    ///
    /// # Returns
    /// A reference to a vector of outputs from each layer in the network.
    ///
    /// # Errors
    /// Returns an [`Error`] if dense block forward propagation fails.
    pub fn forward<'a>(
        &'a self,
        context: &GpuContext,
        input: &'a Tensor2D<T>,
        batch_size: usize,
        use_dropout: bool,
        step: usize,
    ) -> Result<Vec<&'a Tensor2D<T>>, Error> {
        let mut outputs = Vec::with_capacity(self.layers.len());
        let mut current_input = input;

        for layer in &self.layers {
            current_input = layer.forward(context, current_input, batch_size, use_dropout, step)?;
            outputs.push(current_input);
        }

        Ok(outputs)
    }

    /// Executes a sequential forward execution pass.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `raw_pixels` - The input vector for the first layer. This vector will be converted into a tensor internally.
    /// * `batch_size` - The batch size for this pass.
    /// * `use_dropout` - If set to false, dropout is disabled.
    /// * `step` - The current training iteration step.
    ///
    /// # Returns
    /// A **copy** of the output tensor from the very last layer.
    ///
    /// # Errors
    /// Returns an [`Error`] if dense block forward propagation or GPU memory allocation fails.
    pub fn forward_raw(
        &self,
        context: &GpuContext,
        raw_pixels: &[T],
        batch_size: usize,
        use_dropout: bool,
        step: usize,
    ) -> Result<Tensor2D<T>, Error> {
        let input_tensor =
            Tensor2D::from_cpu_vector(context, raw_pixels, &[batch_size, self.inputs])?;

        let all_outputs = self.forward(context, &input_tensor, batch_size, use_dropout, step)?;

        let final_output = all_outputs
            .last()
            .ok_or_else(|| Error::InvalidConfiguration {
                reason: "Cannot execute raw forward pass on an empty network matrix architecture"
                    .to_string(),
            })?;

        (*final_output).clone(context)
    }

    /// Executes a sequential backward execution pass.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `target` - A [`Tensor2D<T>`] with size `(batch size, output features)` representing
    ///   the current target batch.
    /// * `input` - A [`Tensor2D<T>`] that contains the input to this layer during the forward pass.
    /// * `loss_config` - See [`LossConfig`].
    /// * `train_ctx` - Information and hyperparameters for training.
    /// * `step` - The current training iteration step.
    ///
    /// # Returns
    /// A **copy** of the output tensor from the very last layer.
    ///
    /// # Errors
    /// Returns an [`Error`] if dense block back propagation fails.
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        context: &GpuContext,
        outputs: &[&Tensor2D<T>],
        target: &Tensor2D<T>,
        input: &Tensor2D<T>,
        loss_config: &LossConfig,
        train_ctx: &MultiLayerTrainContext,
        step: usize,
    ) -> Result<(), Error> {
        let num_layers = self.layers.len();
        if outputs.len() != num_layers || train_ctx.optimisers.len() != num_layers {
            return Err(Error::InvalidConfiguration {
                reason: format!(
                    "Backward pass expects exactly {num_layers} activation vectors and optimizers. Found {} outputs, {} optimisers.",
                    outputs.len(),
                    train_ctx.optimisers.len()
                ),
            });
        }

        let last_idx = num_layers - 1;
        let head_input = if last_idx > 0 {
            outputs[last_idx - 1]
        } else {
            input
        };

        // Compute output layer
        self.layers[last_idx].compute_loss(context, target, loss_config)?;

        self.layers[last_idx].backward_output(
            context,
            head_input,
            &train_ctx.get_train_context(last_idx),
            step,
        )?;

        // Compute hidden layer
        for i in (0..last_idx).rev() {
            let next_layer = &self.layers[i + 1];
            let layer_input = if i > 0 { outputs[i - 1] } else { input };

            self.layers[i].backward_hidden(
                context,
                next_layer,
                layer_input,
                &train_ctx.get_train_context(i),
                step,
            )?;
        }

        Ok(())
    }

    getter!(pub get_layers, layers, Vec<DenseBlock<T>>);

    /// Returns the total number of layers in the network.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns true if neural network is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}
