use crate::io::device::GpuContext;
use crate::layers::{save_dense_blocks, save_dense_blocks_with_metadata, DenseBlock};
use crate::util::core::Tensor;
use crate::util::log::Error;
use crate::util::precision::PrecisionType;
use std::path::Path;
use crate::util::functions::{Activation, LossFunc, Optimiser};

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
    pub fn new(layers: Vec<DenseBlock<T>>, inputs: usize) -> Self {
        Self {
            layers,
            inputs,
        }
    }

    /// Executes a sequential forward execution pass.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `input` - The input tensor for the first layer.
    /// * `batch_size` - The batch size for this pass.
    /// * `use_dropout` - If set to false, dropout is disabled.
    /// * `step` - The current learning step.
    ///
    /// # Returns
    /// A reference to a vector of outputs from each layer in the network.
    pub fn forward<'a>(
        &'a self,
        context: &GpuContext,
        input: &'a Tensor<T>,
        batch_size: usize,
        use_dropout: bool,
        step: usize,
    ) -> Result<Vec<&'a Tensor<T>>, Error> {
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
    /// * `step` - The current learning step.
    ///
    /// # Returns
    /// A **copy** of the output tensor from the very last layer.
    pub fn forward_raw(
        &self,
        context: &GpuContext,
        raw_pixels: &[T],
        batch_size: usize,
        use_dropout: bool,
        step: usize,
    ) -> Result<Tensor<T>, Error> {
        let input_tensor =
            Tensor::from_cpu_vector(context, raw_pixels, &[batch_size, self.inputs]);

        let all_outputs = self.forward(context, &input_tensor, batch_size, use_dropout, step)?;

        let final_output = all_outputs
            .last()
            .ok_or_else(|| Error::InvalidConfiguration {
                reason: "Cannot execute raw forward pass on an empty network matrix architecture"
                    .to_string(),
            })?;

        Ok((*final_output).clone(context))
    }

    /// Executes a sequential backward execution pass.
    ///
    /// # Arguments
    /// * `context` - GPU Context. See [`GpuContext`].
    /// * `target` - A [`Tensor<T>`] with size `(batch size, output features)` representing
    /// the current target batch.
    /// * `input` - A [`Tensor<T>`] that contains the input to this layer during the forward pass.
    /// * `out_loss_func` - See [`LossFunc`] for the available error functions.
    /// * `out_act_mode` - See [`Activation`] for the available activation functions.
    /// * `batch_size` - Size of the batch.
    /// * `optimisers` - These [`Optimiser`] is used for linear weights and biases.
    /// * `norm_optimiser` - These [`Optimiser`] is used for normalisation weights and biases.
    /// * `learn_rate` - The learning rate of the network. Ideally, it should be between `0.0` exclusive and `1.0` inclusive.
    /// * `clamp` - Clamps gradient values between -`clamp` and +`clamp`. Pass [`f32::MAX`] to turn off clamping.
    /// * `step` - The current learning step.
    ///
    /// # Returns
    /// A **copy** of the output tensor from the very last layer.
    pub fn backward(
        &self,
        context: &GpuContext,
        outputs: &[&Tensor<T>],
        target: &Tensor<T>,
        input: &Tensor<T>,
        out_loss_func: LossFunc,
        out_act_mode: Activation,
        batch_size: usize,
        optimisers: &[Optimiser],
        norm_optimisers: &[Optimiser],
        lr: f32,
        clamp: f32,
        step: usize,
    ) -> Result<(), Error> {
        let num_layers = self.layers.len();
        if outputs.len() != num_layers || optimisers.len() != num_layers {
            return Err(Error::InvalidConfiguration {
                reason: format!(
                    "Backward pass expects exactly {num_layers} activation vectors and optimizers. Found {} outputs, {} optimisers.",
                    outputs.len(),
                    optimisers.len()
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
        self.layers[last_idx].compute_loss(
            context,
            target,
            out_loss_func,
            out_act_mode
        )?;

        self.layers[last_idx].backward_output(
            context,
            head_input,
            batch_size,
            &optimisers[last_idx],
            &norm_optimisers[last_idx],
            lr,
            clamp,
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
                batch_size,
                &optimisers[i],
                &norm_optimisers[i],
                lr,
                clamp,
                step,
            )?;
        }

        Ok(())
    }

    /// Provides a read-only reference to the individual layer components.
    pub fn layers(&self) -> &[DenseBlock<T>] {
        &self.layers
    }

    /// Returns the total number of layers in the network.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Saves the network into a .safetensors file.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `path` - Path of the save file.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::SerializationCasting`] - The underlying tensor raw numerical buffers fail memory alignment
    ///   or size validation thresholds while being transmuted into binary safe arrays via `bytemuck`.
    /// * [`Error::SerdeJSON`] - The provided metadata block cannot be parsed or mapped into valid JSON string parameters.
    /// * [`Error::IOError`] - The system fails to create the file at the specified `path`, hits a storage capacity allocation
    ///   threshold, or encounters an issue flushing the stream to disk.
    pub fn save<P: AsRef<Path>>(&self, context: &GpuContext, path: P) -> Result<(), Error> {
        let layer_refs: Vec<&DenseBlock<T>> = self.layers.iter().collect();
        save_dense_blocks(context, path, &layer_refs)
    }

    /// Saves the network into a .safetensors file.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `path` - Path of the save file.
    /// * `meta` - Metadata to be included inside the save file.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::SerializationCasting`] - The underlying tensor raw numerical buffers fail memory alignment
    ///   or size validation thresholds while being transmuted into binary safe arrays via `bytemuck`.
    /// * [`Error::SerdeJSON`] - The provided metadata block cannot be parsed or mapped into valid JSON string parameters.
    /// * [`Error::IOError`] - The system fails to create the file at the specified `path`, hits a storage capacity allocation
    ///   threshold, or encounters an issue flushing the stream to disk.
    pub fn save_with_metadata<P: AsRef<Path>, K, V>(
        &self,
        context: &GpuContext,
        path: P,
        meta: &[(K, V)],
    ) -> Result<(), Error>
    where
        K: AsRef<str>,
        V: ToString,
    {
        let layer_refs: Vec<&DenseBlock<T>> = self.layers.iter().collect();
        save_dense_blocks_with_metadata(context, path, &layer_refs, meta)
    }
}
