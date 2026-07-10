pub mod io {
    pub mod device;
    pub mod save;
    pub(crate) mod safetensor;
}

pub mod util {
    pub mod scheduler;
    pub mod precision;
    pub mod core;
    pub mod functions;
    pub mod log;
}

pub mod layers;
pub mod ffn;

#[cfg(test)]
mod tests {
    use crate::ffn::FeedForwardNetwork;
    use crate::io::device::GpuContext;
    use crate::layers::DenseBlock;
    use crate::util::core::Tensor;
    use crate::util::functions::{Activation, InitFunc, InitHeUniformFunc, LossFunc, Normalisation, Regularisation};
    use crate::util::functions::Optimiser::SGD;
    use crate::util::log::Error;

    const BATCH_SIZE: usize = 64;

    fn gen_input(context: &GpuContext) -> Tensor<f32> {
        Tensor::zeros(&context, &[64, 32])
    }

    fn gen_target(context: &GpuContext) -> Tensor<f32> {
        Tensor::zeros(&context, &[64, 4])
    }

    #[test]
    fn it_works() {
        example().unwrap();
    }

    fn example() -> Result<(), Error> {
        // 1. Create the GPU context
        let context = GpuContext::new(16); // CUDA tile dimension of 16

        // 2. Create the initialiser
        // The seed 10 is arbitrarily chosen.
        // Factor multiples the weights. A low factor ensures weights are small in the beginning.
        let mut init = InitHeUniformFunc::new::<f32>(10, 0.1);

        // 3. Create the layers. For example, a 32-16-8-4 network.
        // The activation of the last layer (the output layer) must be set to Identity
        let layers: Vec<DenseBlock<f32>> = vec![
            DenseBlock::new(&context, true, 32, 16, BATCH_SIZE, &mut init,
                            Normalisation::Disabled, Activation::LeakyReLU(0.01), Regularisation::None, 0.1),
            DenseBlock::new(&context, true, 16, 8, BATCH_SIZE, &mut init,
                            Normalisation::Disabled, Activation::LeakyReLU(0.01), Regularisation::None, 0.1),
            DenseBlock::new(&context, true, 8, 4, BATCH_SIZE, &mut init,
                            Normalisation::Disabled, Activation::Identity, Regularisation::None, 0.1),
        ];

        // 4. Wrap the layers inside the Feed Forward Network to simplify the process.
        let network = FeedForwardNetwork::<f32>::new(layers, 32);

        // 5. A normal single training loop with forward and backward passes
        // Here, the input tensor takes size (BATCH SIZE, 32) since there are 32 input neurons.
        // Whereas, the target tensor takes size (BATCH SIZE, 4) since there are 4 output neurons.
        let input = &gen_input(&context); // This is the input to your network
        let sgd = SGD(0.9, true); // Stochastic Gradient Descent with Nesterov momentum
        let outputs = network.forward(&context, &input, BATCH_SIZE, true, 1)?;
        network.backward(&context, &outputs, &gen_target(&context), &input,
                         LossFunc::CrossEntropyLoss, Activation::Softmax, BATCH_SIZE,
                         &[sgd, sgd, sgd], &[sgd, sgd, sgd], 0.01, f32::MAX, 1)?;

        Ok(())
    }
}
