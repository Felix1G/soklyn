#![deny(clippy::all)]
#![warn(clippy::pedantic)]

pub mod io {
    #[allow(clippy::similar_names)]
    pub mod device;
    pub(crate) mod safetensor;
    pub mod save;
}

pub mod util {
    pub mod context;
    pub mod core;
    pub mod function;
    pub mod log;
    pub mod scheduler;
    pub mod r#type;
}

pub mod conv;
pub mod ffn;
pub mod mlp;

pub use conv::*;
pub use ffn::*;
pub use mlp::*;
pub use util::function::*;
pub use util::context::*;
pub use util::r#type::*;
pub use util::*;

#[cfg(test)]
mod tests {
    use crate::{LayerInitConfig, LossConfig, MultiLayerTrainContext};
    use crate::ffn::FeedForwardNetwork;
    use crate::io::device::GpuContext;
    use crate::mlp::DenseBlock;
    use crate::util::core::Tensor2D;
    use crate::util::function::Optimiser::SGD;
    use crate::util::function::{
        Activation, InitFunc, InitHeUniformFunc, LossFunc, Normalisation, Regularisation,
    };
    use crate::util::log::Error;
    use std::process::exit;

    const BATCH_SIZE: usize = 64;

    fn gen_input(context: &GpuContext) -> Tensor2D<f32> {
        Tensor2D::zeros(context, &[64, 32]).unwrap()
    }

    fn gen_target(context: &GpuContext) -> Tensor2D<f32> {
        Tensor2D::zeros(context, &[64, 4]).unwrap()
    }

    #[test]
    fn it_works() {
        if let Err(err) = example1() {
            eprintln!("\n[!] Execution Error: {err}");
            exit(1);
        }
    }

    fn example1() -> Result<(), Error> {
        // 1. Create the GPU context
        let context = GpuContext::new(16); // CUDA tile dimension of 16

        // 2. Create the initialiser
        // The seed 10 is arbitrarily chosen.
        // Factor multiples the weights. A low factor ensures weights are small in the beginning.
        let mut init = InitHeUniformFunc::new::<f32>(10, 0.1);

        // 3. Create the layers. For example, a 32-16-8-4 network.
        // The activation of the last layer (the output layer) must be set to Identity
        let layer_config = LayerInitConfig {
            normalisation: Normalisation::Disabled,
            activation: Activation::LeakyReLU { coeff: 0.01 },
            regularisation: Regularisation::None,
            mask_coeff: 0.1,
        };

        let layers: Vec<DenseBlock<f32>> = vec![
            DenseBlock::new_with_config(
                &context,
                true,
                32,
                16,
                BATCH_SIZE,
                &mut init,
                &layer_config,
            )?,
            DenseBlock::new_with_config(
                &context,
                true,
                16,
                8,
                BATCH_SIZE,
                &mut init,
                &layer_config,
            )?,
            DenseBlock::new(&context, true, 8, 4, BATCH_SIZE, &mut init)?,
        ];

        // 4. Wrap the layers inside the Feed Forward Network to simplify the process.
        let network = FeedForwardNetwork::<f32>::new(layers, 32);

        // 5. A normal single training loop with forward and backward passes
        // Here, the input tensor takes size (BATCH SIZE, 32) since there are 32 input neurons.
        // Whereas, the target tensor takes size (BATCH SIZE, 4) since there are 4 output neurons.
        let input = &gen_input(&context); // This is the input to your network
        let sgd = SGD {
            v_coeff: 0.9,
            nesterov: true,
        }; // Stochastic Gradient Descent with Nesterov momentum
        let outputs = network.forward(&context, input, BATCH_SIZE, true, 1)?;

        network.backward(
            &context,
            &outputs,
            &gen_target(&context),
            input,
            &LossConfig {
                loss_func: LossFunc::CrossEntropyLoss,
                activation: Activation::Softmax,
            },
            &MultiLayerTrainContext {
                batch_size: BATCH_SIZE,
                optimisers: &[sgd, sgd, sgd],
                norm_optimisers: &[sgd, sgd, sgd],
                learn_rate: 0.01,
                grad_clamp: f32::MAX,
            },
            1,
        )?;

        Ok(())
    }
}
