use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{RngCore, SeedableRng};
use crate::{getter, getter_copy};
use crate::log::Error;
use crate::util::function::Optimiser::{Adam, SGD};
use crate::util::r#type::PrecisionType;

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
            Self::None => 0,
            Self::L1Regular(_) => 1,
            Self::L2Regular(_) => 2,
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
            Self::Disabled => 0,
            Self::RMSNorm => 1,
            Self::LayerNorm => 2,
            Self::BatchNorm => 3
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
            Self::Identity => 0,
            Self::Sigmoid => 1,
            Self::ReLU => 2,
            Self::LeakyReLU(_) => 3,
            Self::Tanh => 4,
            Self::Softmax => 5,
            Self::SiLU => 6,
            Self::Mish => 7
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
    fn init<T: PrecisionType>(&mut self, fan_in: usize, fan_out: usize, len: usize) -> Vec<T>;
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

    fn init<T: PrecisionType>(&mut self, fan_in: usize, fan_out: usize, len: usize) -> Vec<T> {
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        uniform_dist(len, &mut self.rng, self.factor, limit)
    }
}

impl InitFunc for InitXavierNormalFunc {
    fn new<T: PrecisionType>(seed: u64, factor: f32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            factor
        }
    }

    fn init<T: PrecisionType>(&mut self, fan_in: usize, fan_out: usize, len: usize) -> Vec<T> {
        let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
        normal_dist(len, &mut self.rng, self.factor, std)
    }
}
impl InitFunc for InitHeUniformFunc {
    fn new<T: PrecisionType>(seed: u64, factor: f32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            factor
        }
    }

    fn init<T: PrecisionType>(&mut self, fan_in: usize, _: usize, len: usize) -> Vec<T> {
        let limit = (6.0 / fan_in as f32).sqrt();
        uniform_dist(len, &mut self.rng, self.factor, limit)
    }
}

impl InitFunc for InitHeNormalFunc {
    fn new<T: PrecisionType>(seed: u64, factor: f32) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            factor
        }
    }

    fn init<T: PrecisionType>(&mut self, fan_in: usize, _: usize, len: usize) -> Vec<T> {
        let std = (2.0 / fan_in as f32).sqrt();
        normal_dist(len, &mut self.rng, self.factor, std)
    }
}

impl InitFunc for InitZeroFunc {
    /// `_seed` is Redundant. You may leave it as any number.
    fn new<T: PrecisionType>(_seed: u64, _factor: f32) -> Self {
        Self {}
    }

    fn init<T: PrecisionType>(&mut self, _: usize, _: usize, len: usize) -> Vec<T> {
        vec![T::zero(); len]
    }
}

pub enum PaddingType {
    /// Padding is filled with `0`s.
    ///
    /// `0 0 | A B C | 0 0`
    ZeroPadding,
    /// Padding pixels are a mirror image of the pixels inside the edge.
    ///
    /// `C B | A B C | B A`
    ReflectivePadding,
    /// Takes the very last pixel on the edge and repeats it outwards indefinitely.
    ///
    /// `A A | A B C | C C`
    ReplicatePadding,
}

impl PaddingType {
    pub(crate) fn ordinal(&self) -> usize {
        match self {
            Self::ZeroPadding => 0,
            Self::ReflectivePadding => 1,
            Self::ReplicatePadding => 2,
        }
    }
}

pub struct KernelConfig {
    dimension: (usize, usize),
    pad: usize,
    auto_pad: bool,
    pad_type: PaddingType,
    stride: (usize, usize),
    dilation: (usize, usize),
}

impl KernelConfig {
    /// Creates a manual padding configuration where an explicit number of zero-filled rows
    /// and columns are symmetrically appended to all outer boundaries of the input tensor.
    ///
    /// # Arguments
    /// * `dimension` - Spatial width (`dimension.0`) and height (`dimension.1`) of each kernel.
    /// * `pad` - The number of padding layers to apply symmetrically along both spatial dimensions (width and height).
    /// * `pad_type` - The type of padding to use. See [`PaddingType`].
    /// * `stride` - Discrete step increment for the sliding window.
    /// `stride.0` represents the x-axis (width) while `stride.1` represents the y-axis (height).
    /// * `dilation` - The spacing between kernel elements, inserting `dilation - 1` spaces in between.
    pub fn new(dimension: (usize, usize), pad: usize, pad_type: PaddingType,
               stride: (usize, usize), dilation: (usize, usize)) -> Result<Self, Error> {
        if dimension.0 == 0 {
            return Err(Error::InvalidConfiguration {
                reason: String::from("Kernel width cannot be 0."),
            });
        }

        if dimension.1 == 0 {
            return Err(Error::InvalidConfiguration {
                reason: String::from("Kernel height cannot be 0."),
            });
        }

        if stride.0 == 0 || stride.1 == 0 {
            return Err(Error::InvalidConfiguration {
                reason: String::from("Kernel stride cannot be 0."),
            });
        }

        if stride.0 > dimension.0 {
            log::warn!("Kernel stride is more than the width. Some data will be ignored!");
        }

        if stride.1 > dimension.1 {
            log::warn!("Kernel stride is more than the height. Some data will be ignored!");
        }

        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(Error::InvalidConfiguration {
                reason: String::from("Kernel dilation cannot be 0."),
            });
        }

        Ok(Self {
            dimension,
            pad,
            auto_pad: false,
            pad_type,
            stride,
            dilation
        })
    }

    /// Creates an automatic padding configuration that dynamically computes the required
    /// padding size at execution time, ensuring that the filter covers all spatial boundary
    /// elements and the output dimensions are ceiling-divided by the stride value.
    pub fn auto_pad(dimension: (usize, usize), pad_type: PaddingType,
                    stride: (usize, usize), dilation: (usize, usize)) -> Result<Self, Error> {
        Self::new(dimension, 0, pad_type, stride, dilation)
    }

    getter!(pub get_dimension, dimension, (usize, usize));
    getter_copy!(pub get_pad, pad, usize);
    getter!(pub get_pad_type, pad_type, PaddingType);
    getter!(pub get_stride, stride, (usize, usize));
    getter!(pub get_dilation, dilation, (usize, usize));

    pub(crate) fn auto_pad_val(
        &self,
        dim: &(usize, usize),
    ) -> (usize, usize) {
        if self.auto_pad {
            let out = self.elements_from_length(dim);
            let needed_w = (out.0 - 1) * self.stride.0 + self.dimension.0;
            let pad_w = if needed_w > dim.0 {
                needed_w - dim.0
            } else {
                0
            };

            let needed_h = (out.1 - 1) * self.stride.1 + self.dimension.1;
            let pad_h = if needed_h > dim.1 {
                needed_h - dim.1
            } else {
                0
            };

            (pad_w, pad_h)
        } else {
            (0, 0)
        }
    }

    #[inline]
    pub(crate) fn actual_width(&self) -> usize {
        (self.dimension.0 - 1) * self.dilation.0 + 1
    }

    #[inline]
    pub(crate) fn actual_height(&self) -> usize {
        (self.dimension.1 - 1) * self.dilation.1 + 1
    }

    /// # Arguments
    /// * `len` - The dimension of the input (spatial width, spatial height).
    pub(crate) fn elements_from_length(&self, len: &(usize, usize)) -> (usize, usize) {
        if self.auto_pad {
            (
                (len.0 + self.stride.0 - 1) / self.stride.0,
                (len.1 + self.stride.1 - 1) / self.stride.1
            )
        } else {
            let padded_len_x = len.0 + 2 * self.pad;
            let padded_len_y = len.1 + 2 * self.pad;

            let size_x = self.actual_width();
            let size_y = self.actual_height();

            let elems_x = if padded_len_x < size_x {
                0
            } else {
                ((padded_len_x - size_x) / self.stride.0) + 1
            };

            let elems_y = if padded_len_y < size_y {
                0
            } else {
                ((padded_len_y - size_y) / self.stride.1) + 1
            };

            (elems_x, elems_y)
        }
    }
}