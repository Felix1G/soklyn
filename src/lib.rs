pub mod nn;
pub mod util;
pub mod device;

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use cudarc::driver::CudaSlice;
    use rand::{rng, RngExt};
    use crate::device::GpuContext;
    use crate::nn::functions::{Activation, InitHeNormalFunc, Regularisation};
    use crate::nn::functions::Activation::{Identity, LeakyReLU, Softmax};
    use crate::nn::functions::ErrorFunc::CrossEntropyLoss;
    use crate::nn::functions::Normalisation::{BatchNorm, Disabled, LayerNorm, RMSNorm};
    use crate::nn::functions::Optimiser::Adam;
    use crate::nn::InitFunc;
    use crate::nn::network::{load_tensors, save_tensors_with_metadata, DenseBlock};
    use crate::util::Tensor;

    fn softmax(mut logits: Vec<f32>, batch_dim: usize, num_classes: usize) -> Vec<f32> {
        assert_eq!(
            logits.len(),
            batch_dim * num_classes,
            "Logits vector length does not match batch_dim * num_classes!"
        );

        // Process each batch row independently
        for b in 0..batch_dim {
            let row_start = b * num_classes;
            let row_end = row_start + num_classes;
            let row_slice = &mut logits[row_start..row_end];

            // 1. Find the maximum value in this specific batch row for numerical stability
            let max_logit = row_slice
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            // 2. Compute exponents for this row and sum them up
            let mut sum_exps = 0.0f32;
            for val in row_slice.iter_mut() {
                *val = (*val - max_logit).exp();
                sum_exps += *val;
            }

            // 3. Divide by the sum to turn exponents into probabilities
            for val in row_slice.iter_mut() {
                *val /= sum_exps;
            }
        }

        logits
    }

    fn download_cuda(context: &GpuContext, c: &CudaSlice<f32>) -> Vec<f32> {
        context.get_stream().clone_dtoh(c).unwrap()
    }

    #[test]
    fn it_works() {
        let context = GpuContext::new(16);
        let batch_size = 5;
        let mut rand = InitHeNormalFunc::new(108, 1.0);

        let layer1 = DenseBlock::new(&context, 2, 4, batch_size, &mut rand, Disabled, LeakyReLU(0.1), Regularisation::L2Regular(0.01), 0.0);
        let layer2 = DenseBlock::new(&context, 4, 4, batch_size, &mut rand, Disabled, LeakyReLU(0.1), Regularisation::L2Regular(0.01), 0.0);
        let layer3 = DenseBlock::new(&context, 4, 2, batch_size, &mut rand, Disabled, Identity, Regularisation::None, 0.0);

        /*let mut layers = load_tensors(&context, "network1.safetensors", batch_size);
        (&mut layers[0]).set_activation(LeakyReLU(0.1));
        (&mut layers[1]).set_activation(LeakyReLU(0.1));
        (&mut layers[2]).set_activation(Identity);
        let layer1 = &layers[0];
        let layer2 = &layers[1];
        let layer3 = &layers[2];*/

        let mut rng = rng();

        for step in 1..=30000 {
            let mut input = Vec::<f32>::new();
            let mut target = Vec::<f32>::new();
            for _ in 0..5 {
                let a = rng.random_range(0.0f32..10.0f32) * 0.1;
                let b = rng.random_range(0.0f32..10.0f32) * 0.1;
                input.push(a); input.push(b);
                if a > b { target.push(0.0); target.push(1.0); }
                else { target.push(1.0); target.push(0.0); }
            }

            let input_tensor = Tensor::from_cpu_vector(&context, &input, &vec![batch_size, 2]);
            let target_tensor = Tensor::from_cpu_vector(&context, &target, &vec![batch_size, 2]);

            let out1 = layer1.forward(&context, &input_tensor, batch_size, true, step);
            let out2 = layer2.forward(&context, &out1, batch_size, true, step);
            layer3.forward(&context, &out2, batch_size, true, step);

            layer3.compute_loss(&context, &target_tensor, CrossEntropyLoss, Softmax);

            let adam = Adam(0.9, 0.999, 0.00000001);
            let grad2 = layer3.backward_output(&context, out2, batch_size, &adam, &adam,0.03, 1.0, step);
            let grad1 = layer2.backward_hidden(&context, grad2, layer3.get_weights(), out1, batch_size, &adam, &adam,0.03, 1.0, step);
            let _grad = layer1.backward_hidden(&context, grad1, layer2.get_weights(), &input_tensor, batch_size, &adam, &adam,0.03, 1.0, step);
        }

        let mut input_v = vec![2.0, 3.0,
                               0.1, 4.0,
                               10.0, 4.0,
                               1.0, 1.0,
                               9.0, 1.5];
        for i in 0..10 {
            input_v[i] *= 0.1;
        }

        println!("{:?}", input_v);
        let input = Tensor::from_cpu_vector(&context, &input_v, &vec![batch_size, 2]);
        let out1 = layer1.forward(&context, &input, batch_size, false, 1);
        let out2 = layer2.forward(&context, &out1, batch_size, false, 1);
        let out3 = layer3.forward(&context, &out2, batch_size, false, 1);
        println!("layer1w: {:?}", layer1.get_weights().download(&context));
        println!("layer1b: {:?}", layer1.get_biases().download(&context));
        println!("layer2w: {:?}", layer2.get_weights().download(&context));
        println!("layer2b: {:?}", layer2.get_biases().download(&context));
        println!("layer3w: {:?}", layer3.get_weights().download(&context));
        println!("layer3b: {:?}", layer3.get_biases().download(&context));
        println!("activations: {:?}", vec![layer1.get_activation(), layer2.get_activation(), layer3.get_activation()]);
        println!("normalisation: {:?}", vec![layer1.get_normalisation(), layer2.get_normalisation(), layer3.get_normalisation()]);
        println!("{:?}", softmax(out3.download(&context).v, batch_size, 2));

        //HeNormal 2-4-4-2 108seed
        //disabled: [-0.17725684, 0.10833322, -0.18628727, 0.109780684, -0.37935716, 0.24348314, -0.06558628, 0.04061284, -0.24464451, 0.16220902]
        //rmsnorm: [-0.2244776, 0.1371929, -0.21951966, 0.12936482, -0.22951795, 0.1473117, -0.22603786, 0.1399689, -0.23145396, 0.15346317]
        //layernorm: [-0.33146033, 0.16285522, -0.3300621, 0.16126384, -0.3459834, 0.1829581, -0.3323434, 0.16388358, -0.35035405, 0.19557093]
        //batchnorm: [0.5843828, -0.8866035, -0.30189195, 0.31706, -0.4815282, 0.34416974, -0.29027864, 0.29446065, 1.3053284, -1.6291916]
        //[0.99999976, 1.8562618e-7, 1.0, 9.63e-43, 0.0, 1.0, 0.57343274, 0.4265672, 0.0, 1.0]
        save_tensors_with_metadata(&context, "network1.safetensors", &vec![&layer1, &layer2, &layer3], &[("a", "b")]);
    }
}
