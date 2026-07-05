use std::collections::HashSet;
use std::process::exit;
use std::time::SystemTime;
use image::{ImageBuffer, Rgb};
use mnist::{Mnist, MnistBuilder};
use rand::prelude::SliceRandom;
use rand::rng;
use intelligent_artificial::device::GpuContext;
use intelligent_artificial::nn::functions::Activation::{Identity, LeakyReLU, Mish};
use intelligent_artificial::nn::functions::{Activation, ErrorFunc, InitHeNormalFunc, InitXavierNormalFunc, Normalisation};
use intelligent_artificial::nn::functions::Normalisation::{BatchNorm, Disabled, LayerNorm};
use intelligent_artificial::nn::functions::Optimiser::Adam;
use intelligent_artificial::nn::InitFunc;
use intelligent_artificial::nn::network::{save_tensors_with_metadata, DenseBlock};
use intelligent_artificial::nn::scheduler::{ExponentialLR, SchedulerMode};
use intelligent_artificial::util::Tensor;

const BATCH_SIZE: usize = 200;
const CLAMP: f32 = f32::MAX;
const EPSILON: f32 = 1e-8;
const TRAIN_ELEMENTS: u32 = 60_000;
const TEST_ELEMENTS: usize = 10_000;

fn main() {
    println!("--- Loading Fashion-MNIST Dataset ---");
    let mnist = MnistBuilder::new()
        .use_fashion_data()
        .label_format_one_hot()
        .training_set_length(TRAIN_ELEMENTS)
        .test_set_length(TEST_ELEMENTS as u32)
        .download_and_extract()
        .finalize();

    let context = GpuContext::new(16);
    let mut rand = InitXavierNormalFunc::new(108, 0.5);

    let mut layers = vec![
        DenseBlock::default(&context, 784, 512, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 512, 256, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 256, 256, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 256, 256, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 256, 512, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 512, 784, BATCH_SIZE, &mut rand),
    ];

    configure_layers(&mut layers);

    let mut scheduler = ExponentialLR::new(0.002, 0.95);

    for n in 1..=70 {
        epoch(n, &mnist, &mut layers, &context, &mut scheduler);
    }
}

fn configure_layers(layers: &mut Vec<DenseBlock>) {
    // Encoder
    layers[0].set_normalisation(BatchNorm);
    layers[0].set_activation(Mish);
    layers[0].set_mask_coeff(0.1);

    layers[1].set_normalisation(BatchNorm);
    layers[1].set_activation(Mish);
    layers[1].set_mask_coeff(0.1);

    // Bottleneck
    layers[2].set_normalisation(LayerNorm);
    layers[2].set_activation(Mish);
    layers[2].set_mask_coeff(0.0);

    layers[3].set_normalisation(LayerNorm);
    layers[3].set_activation(Mish);
    layers[3].set_mask_coeff(0.1);

    // Decoder
    layers[4].set_normalisation(Disabled);
    layers[4].set_activation(Mish);
    layers[4].set_mask_coeff(0.1);

    layers[5].set_normalisation(Disabled);
    layers[5].set_activation(Identity);
    layers[5].set_mask_coeff(0.0);
}

fn epoch(epoch: usize, mnist: &Mnist, layers: &mut Vec<DenseBlock>, context: &GpuContext, scheduler: &mut ExponentialLR) {
    let step_offset = (epoch - 1) * TRAIN_ELEMENTS as usize / BATCH_SIZE;

    println!("--- Epoch #{epoch} ---");
    train(mnist, layers, context, step_offset, scheduler);
    test(mnist, layers, context, epoch);
    println!("--- Epoch #{epoch} complete ---");

    if epoch % 10 == 0 {
        let refs: Vec<&DenseBlock> = layers.iter().collect();
        save_tensors_with_metadata(
            context,
            format!("data/mnist{}AEC.safetensors", epoch / 10),
            &refs,
            &[("epoch", epoch.to_string())],
        );
    }
}

fn forward_all<'a>(
    layers: &'a Vec<DenseBlock>, context: &GpuContext,
    input: &'a Tensor, batch_size: usize, is_training: bool, step: usize
) -> [&'a Tensor; 6] {
    let o1 = layers[0].forward(context, input,  batch_size, is_training, step);
    let o2 = layers[1].forward(context, o1,     batch_size, is_training, step);
    let o3 = layers[2].forward(context, o2,     batch_size, is_training, step);
    let o4 = layers[3].forward(context, o3,     batch_size, is_training, step);
    let o5 = layers[4].forward(context, o4,     batch_size, is_training, step);
    let o6 = layers[5].forward(context, o5,     batch_size, is_training, step);
    [o1, o2, o3, o4, o5, o6]
}

fn train(
    mnist: &Mnist, layers: &Vec<DenseBlock>, context: &GpuContext,
    step_offset: usize, scheduler: &mut ExponentialLR,
) {
    let mut indices: Vec<usize> = (0..mnist.trn_img.len() / 784).collect();
    indices.shuffle(&mut rng());

    let adam = Adam(0.9, 0.999, EPSILON);
    let total_batches = indices.chunks(BATCH_SIZE).len();
    let mut forw_ms = 0.0f64;
    let mut loss_ms = 0.0f64;
    let mut back_ms = 0.0f64;

    for (batch_idx, batch_ids) in indices.chunks(BATCH_SIZE).enumerate() {
        let step = batch_idx + 1 + step_offset;
        let lr = scheduler.get_learning_rate();
        let (input, _) = build_batch(context, &mnist.trn_img, &mnist.trn_lbl, batch_ids, BATCH_SIZE, 784, 10);

        let t0 = SystemTime::now();
        let outs = forward_all(layers, context, &input, BATCH_SIZE, true, step);
        forw_ms += t0.elapsed().unwrap().as_secs_f64() * 1000.0;

        let t1 = SystemTime::now();
        layers[5].compute_loss(context, &input, ErrorFunc::BinaryCrossEntropy, Activation::Sigmoid);
        loss_ms += t1.elapsed().unwrap().as_secs_f64() * 1000.0;

        let t2 = SystemTime::now();
        let g5 = layers[5].backward_output(context, outs[4], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        let g4 = layers[4].backward_hidden(context, g5, layers[5].get_weights(), outs[3], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        let g3 = layers[3].backward_hidden(context, g4, layers[4].get_weights(), outs[2], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        let g2 = layers[2].backward_hidden(context, g3, layers[3].get_weights(), outs[1], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        let g1 = layers[1].backward_hidden(context, g2, layers[2].get_weights(), outs[0], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        layers[0].backward_hidden(context, g1, layers[1].get_weights(), &input, BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        back_ms += t2.elapsed().unwrap().as_secs_f64() * 1000.0;

        assert_no_nan(&outs[5].download(context).v, &format!("training batch {batch_idx}"));
    }

    let avg = |ms: f64| ms / total_batches as f64;
    println!(
        "Forward: {:.2}ms | Loss: {:.2}ms | Backward: {:.2}ms | Total: {:.2}ms | LR: {}",
        avg(forw_ms), avg(loss_ms), avg(back_ms),
        avg(forw_ms + loss_ms + back_ms),
        scheduler.get_learning_rate()
    );

    scheduler.step();
}

fn test(mnist: &Mnist, layers: &Vec<DenseBlock>, context: &GpuContext, epoch: usize) {
    // Save one reconstruction per class (0-9) for visual inspection
    let mut unseen_classes: HashSet<usize> = (0..10).collect();

    for (batch_idx, batch_ids) in (0..TEST_ELEMENTS).collect::<Vec<_>>().chunks(BATCH_SIZE).enumerate() {
        let (input, labels) = build_batch(context, &mnist.tst_img, &mnist.tst_lbl, batch_ids, BATCH_SIZE, 784, 10);
        let outs = forward_all(layers, context, &input, BATCH_SIZE, false, batch_idx);
        let out_mat = outs[5].download(context);

        assert_no_nan(&out_mat.v, &format!("testing batch {batch_idx}"));

        if !unseen_classes.is_empty() {
            let class = labels[0..10].iter().position(|&v| v == 1.0).unwrap();
            if unseen_classes.remove(&class) {
                save_reconstruction(&out_mat.v, &input.download(context).v, class, epoch);
            }
        }
    }
}

fn save_reconstruction(output: &[f32], input: &[f32], class: usize, epoch: usize) {
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let to_pixel = |v: f32| (v * 255.5) as u8;

    // Save reconstruction
    let mut img = ImageBuffer::new(28, 28);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let val = to_pixel(sigmoid(output[y as usize * 28 + x as usize]));
        *pixel = Rgb([val, val, val]);
    }
    img.save(format!("data/{class}_recon.png")).unwrap();

    // Save original
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let val = to_pixel(input[y as usize * 28 + x as usize]);
        *pixel = Rgb([val, val, val]);
    }
    img.save(format!("data/{class}_real.png")).unwrap();
}

fn build_batch(
    context: &GpuContext, imgs: &[u8], lbls: &[u8],
    ids: &[usize], batch_size: usize, img_size: usize, label_size: usize,
) -> (Tensor, Vec<f32>) {
    let mut pixels = Vec::with_capacity(ids.len() * img_size);
    let mut labels = Vec::with_capacity(ids.len() * label_size);
    for &id in ids {
        pixels.extend(imgs[id * img_size..(id + 1) * img_size].iter().map(|&p| p as f32 / 255.0));
        labels.extend(lbls[id * label_size..(id + 1) * label_size].iter().map(|&l| l as f32));
    }
    (Tensor::from_cpu_vector(context, &pixels, &vec![batch_size, img_size]), labels)
}

fn assert_no_nan(values: &[f32], context: &str) {
    if values.iter().any(|v| v.is_nan()) {
        eprintln!("NAN DETECTED: {context}");
        exit(1);
    }
}