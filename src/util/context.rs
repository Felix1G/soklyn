use crate::{Activation, LossFunc, Normalisation, Optimiser, Regularisation};

/// Initialisation config for layers. Pass this to layer construction functions.
///
/// # Fields
/// * `normalisation` - See [`Normalisation`]. Pass [`Disabled`] to disable normalisation.
/// * `activation` - See [`Activation`]. Pass [`Identity`] to set this layer to an output layer.
/// * `regularisation` - See [`Regularisation`]. Pass [`Regularisation::None`] to disable regularisation.
/// * `mask_coeff` - Mask coefficient for dropout.
#[derive(Debug, Clone, Copy)]
pub struct LayerInitConfig {
    pub normalisation: Normalisation,
    pub activation: Activation,
    pub regularisation: Regularisation,
    pub mask_coeff: f32,
}

impl Default for LayerInitConfig {
    fn default() -> Self {
        Self {
            normalisation: Normalisation::Disabled,
            activation: Activation::Identity,
            regularisation: Regularisation::None,
            mask_coeff: 0.0,
        }
    }
}

/// Pass this context to backward propagation functions.
///
/// # Fields
/// * `optimiser` - This [`Optimiser`] is used for linear weights and biases.
/// * `norm_optimiser` - This [`Optimiser`] is used for normalisation weights and biases.
/// * `batch_size` - Size of the batch.
/// * `learn_rate` - The learning rate of the network. Ideally, it should be between `0.0` exclusive and `1.0` inclusive.
/// * `grad_clamp` - Clamps gradient values between `-grad_clamp` and `+grad_clamp`. Pass [`f32::MAX`] to turn off clamping.
/// * `step` - The current training iteration step.
#[derive(Debug, Clone, Copy)]
pub struct TrainContext<'a> {
    pub optimiser: &'a Optimiser,
    pub norm_optimiser: &'a Optimiser,
    pub batch_size: usize,
    pub learn_rate: f32,
    pub grad_clamp: f32,
}

/// Pass this context to backward propagation functions.
///
/// # Fields
/// * `optimisers` - The [`Optimiser`]s used for linear weights and biases in each layer.
/// * `norm_optimisers` - The [`Optimiser`]s used for normalisation weights and biases in each layer.
/// * `batch_size` - Size of the batch.
/// * `learn_rate` - The learning rate of the network. Ideally, it should be between `0.0` exclusive and `1.0` inclusive.
/// * `grad_clamp` - Clamps gradient values between `-grad_clamp` and `+grad_clamp`. Pass [`f32::MAX`] to turn off clamping.
#[derive(Debug, Clone, Copy)]
pub struct MultiLayerTrainContext<'a> {
    pub optimisers: &'a [Optimiser],
    pub norm_optimisers: &'a [Optimiser],
    pub batch_size: usize,
    pub learn_rate: f32,
    pub grad_clamp: f32,
}

impl MultiLayerTrainContext<'_> {
    pub(crate) fn get_train_context(&'_ self, idx: usize) -> TrainContext<'_> {
        TrainContext {
            optimiser: &self.optimisers[idx],
            norm_optimiser: &self.norm_optimisers[idx],
            batch_size: self.batch_size,
            learn_rate: self.learn_rate,
            grad_clamp: self.grad_clamp,
        }
    }
}

/// Configuration for output loss computation.
///
/// # Fields
/// * `loss_func` - The loss function to be used. See [`LossFunc`] for available loss functions.
/// * `act_mode` - The outputs's final [`Activation`] (e.g. [`Activation::Softmax`]).
#[derive(Debug, Clone, Copy)]
pub struct LossConfig {
    pub loss_func: LossFunc,
    pub activation: Activation,
}
