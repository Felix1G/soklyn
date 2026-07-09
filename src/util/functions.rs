use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use crate::util::functions::Optimiser::{Adam, SGD};
use crate::util::precision::PrecisionType;

/// Regularisation is applied right before the optimiser.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Regularisation {
    /// No regularisation is used.
    None,
    /// L1 Regularisation (Lasso). This adds a penalty proportional to the absolute value of the
    /// weights in order to drive less important weights directly into `0.0`, which creates a
    /// simpler and sparser model.
    ///
    /// The derivative formula is `w[i] -= L * sign(w[i])`.
    ///
    /// `L` is the regularisation coefficient.
    L1Regular(f32),
    /// L2 Regularisation (Ridge). This adds a penalty proportional to the squared values of the
    /// weights, reducing them close to `0.0` without completely eliminating them. This is to
    /// prevent overfitting.
    ///
    /// The derivative formula is `w[i] -= 2 * L * w[i]`.
    ///
    /// `L` is the regularisation coefficient.
    L2Regular(f32),
}

impl Regularisation {
    /// IMPORTANT: 0 is for NO regularisation.
    pub(crate) fn ordinal(&self) -> usize {
        match self {
            Regularisation::None => 0,
            Regularisation::L1Regular(_) => 1,
            Regularisation::L2Regular(_) => 2,
        }
    }
}

/// Normalization methods used to scale hidden layer activations. They
/// improve training stability and convergence efficiency.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Normalisation {
    /// Features pass without normalisation.
    Disabled,
    /// Scales activations by their root-mean-square.
    ///
    /// Applied across the feature dimension for each sample independently.
    RMSNorm,
    /// Scales activations by their mean and variance.
    ///
    /// Applied across the feature dimension for each sample independently.
    LayerNorm,
    /// Scales activations across the entire batch dimension. For BatchNorm, linear biases are
    /// not considered i.e. removed from calculation due to the linear bias being eliminated from the mathematical formula.
    ///
    /// Applied to each individual feature independently across all samples in a batch.
    BatchNorm
}

impl Normalisation {
    /// IMPORTANT: 0 is for NO normalisation.
    pub(crate) fn ordinal(&self) -> usize {
        match self {
            Normalisation::Disabled => 0,
            Normalisation::RMSNorm => 1,
            Normalisation::LayerNorm => 2,
            Normalisation::BatchNorm => 3
        }
    }
}

/// Optimisers are rules that dictate how weights and biases are updated.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Optimiser {
    /// The optimiser that updates model parameters by taking a single step in the
    /// exact opposite direction of the current batch's gradient, scaled by a learning rate.
    ///
    /// # Parameters
    /// * `momentum coefficient` - The amount of 'friction' on the momentum. If set to `0.0`,
    /// momentum is turned off and no data for momentum will be stored.
    /// * `nesterov` - If set to true, uses Nesterov Accelerated Gradient.
    SGD(f32, bool),

    /// Adaptive optimisation algorithm that speeds up training by calculating
    /// unique learning rates for individual parameters based on a running combination of
    /// their average gradient direction (momentum) and their gradient variance (scaling factor).
    ///
    /// # Parameters
    /// * `first moment coefficient` - The amount of 'friction' on the momentum.
    /// * `second moment coefficient` - Controls how quickly past gradient magnitudes adapt.
    /// * `epsilon` - A very small value to prevent zero division.
    Adam(f32, f32, f32)
}

impl Optimiser {
    pub(crate) fn ordinal(&self) -> usize {
        match self {
            SGD(_, _) => 0,
            Adam(_, _, _) => 1,
        }
    }
}

/// Supported mathematical activation functions.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Activation {
    /// All values pass through normally.
    Identity,
    /// Squeezes any real-valued input into a smooth, continuous range between **0.0 and 1.0**.
    Sigmoid,
    /// Passes positive values through completely unchanged, while mapping all negative values
    /// strictly to **0.0**.
    ///
    /// Recommended to use He initialisations.
    ReLU,
    /// Passes positive values through completely unchanged, while multiplying all negative values
    /// by `alpha`.
    ///
    /// Recommended to use He initialisations.
    ///
    /// # Parameters
    /// * `alpha` - LeakyReLU negative coefficient (`alpha * x`).
    LeakyReLU(f32),
    /// Squeezes any real-valued input into a smooth, continuous range between **-1.0 and 1.0**.
    Tanh,
    /// Converts the logits into a probability distribution, ensuring all outputs range between '0.0' and '1.0' (inclusive) and sum up to exactly '1.0'.
    Softmax,
    /// Multiplies the input by its own sigmoid value to create a smooth, non-monotonic variant of ReLU that avoids dying neurons.
    /// Recommended to use Xavier initialisations and a lower learning rate.
    SiLU,
    /// Multiplies the input by the hyperbolic tangent of its softplus transformation, offering a deeply smooth negative pocket for advanced gradient flow.
    /// Recommended to use Xavier initialisations and a lower learning rate.
    Mish
}

impl Activation {
    pub fn ordinal(&self) -> usize {
        match self {
            Activation::Identity => 0,
            Activation::Sigmoid => 1,
            Activation::ReLU => 2,
            Activation::LeakyReLU(_) => 3,
            Activation::Tanh => 4,
            Activation::Softmax => 5,
            Activation::SiLU => 6,
            Activation::Mish => 7
        }
    }
}

/// Supported error calculation functions.
#[derive(Debug, Clone, Copy)]
pub enum LossFunc {
    /// Cannot be paired with softmax. (`1/2N * (target - output)^2`)
    MeanSquareLoss,
    /// Can only be paired with softmax.
    CrossEntropyLoss,
    /// Can only be paired with sigmoid
    BinaryCrossEntropy,
}

/// Used for initialising initial network parameters.
///
/// # Arguments for `new`
/// * `seed` - Seed for the random number generator used.
/// * `factor` - A factor constant to be multiplied with the generated initial weight values. (e.g. `1.0`, `0.1`)
///
/// # Classes
/// * [`InitXavierUniformFunc`] - Initialises weights from a uniform distribution scaled by the sum of input and output dimensions, ideal for symmetric activations like Tanh.
/// * [`InitXavierNormalFunc`] - Initialises weights from a zero-mean normal distribution scaled by the sum of input and output dimensions, ideal for symmetric activations like Tanh.
/// * [`InitHeUniformFunc`] - Initialises weights from a uniform distribution scaled strictly by the input dimensions to preserve signal variance when using ReLU activations.
/// * [`InitHeNormalFunc`] - Initialises weights from a zero-mean normal distribution scaled strictly by the input dimensions to preserve signal variance when using ReLU activations.
/// * [`InitZeroFunc`] - Initialises to `0.0`.
pub trait InitFunc {
    fn new<T: PrecisionType>(seed: u64, mul: f32) -> Self;
    fn init<T: PrecisionType>(&mut self, fan_in: usize, fan_out: usize) -> Vec<T>;
}

pub struct InitXavierUniformFunc {
    rng: ChaCha8Rng,
    factor: f32
}

pub struct InitXavierNormalFunc {
    rng: ChaCha8Rng,
    factor: f32
}

pub struct InitHeUniformFunc {
    rng: ChaCha8Rng,
    factor: f32
}

pub struct InitHeNormalFunc {
    rng: ChaCha8Rng,
    factor: f32
}

/// Initialises to `0.0`.
pub struct InitZeroFunc {}

fn normal_dist<T: PrecisionType>(total: usize, rng: &mut ChaCha8Rng, factor: f32, std: f32) -> Vec<T> {
    (0..total)
        .map(|_| {
            let u1 = (rng.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 1.0);
            let u2 = rng.next_u32() as f32 / u32::MAX as f32;
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            T::from_f32(factor * z * std)
        })
        .collect()
}

fn uniform_dist<T: PrecisionType>(total: usize, rng: &mut ChaCha8Rng, factor: f32, limit: f32) -> Vec<T> {
    (0..total)
        .map(|_| {
            let raw = rng.next_u32() as f32 / u32::MAX as f32;
            T::from_f32(factor * (-limit + raw * (2.0 * limit)))
        })
        .collect()
}

impl InitFunc for InitXavierUniformFunc {
    fn new<T: PrecisionType>(seed: u64, factor: f32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            factor
        }
    }

    fn init<T: PrecisionType>(&mut self, fan_in: usize, fan_out: usize) -> Vec<T> {
        let total = fan_in * fan_out;
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        uniform_dist(total, &mut self.rng, self.factor, limit)
    }
}

impl InitFunc for InitXavierNormalFunc {
    fn new<T: PrecisionType>(seed: u64, factor: f32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            factor
        }
    }

    fn init<T: PrecisionType>(&mut self, fan_in: usize, fan_out: usize) -> Vec<T> {
        let total = fan_in * fan_out;
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
        normal_dist(total, &mut self.rng, self.factor, std)
    }
}
impl InitFunc for InitHeUniformFunc {
    fn new<T: PrecisionType>(seed: u64, factor: f32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            factor
        }
    }

    fn init<T: PrecisionType>(&mut self, fan_in: usize, fan_out: usize) -> Vec<T> {
        let total = fan_in * fan_out;
        let limit = (6.0 / fan_in as f32).sqrt();
        uniform_dist(total, &mut self.rng, self.factor, limit)
    }
}

impl InitFunc for InitHeNormalFunc {
    fn new<T: PrecisionType>(seed: u64, factor: f32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            factor
        }
    }

    fn init<T: PrecisionType>(&mut self, fan_in: usize, fan_out: usize) -> Vec<T> {
        let total = fan_in * fan_out;
        let std = (2.0 / fan_in as f32).sqrt();
        normal_dist(total, &mut self.rng, self.factor, std)
    }
}

impl InitFunc for InitZeroFunc {
    /// `_seed` is Redundant. You may leave it as any number.
    fn new<T: PrecisionType>(_seed: u64, _factor: f32) -> Self {
        Self {}
    }

    fn init<T: PrecisionType>(&mut self, fan_in: usize, fan_out: usize) -> Vec<T> {
        vec![T::zero(); fan_in * fan_out]
    }
}