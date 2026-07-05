use std::collections::HashSet;
#[allow(unused_imports)]
use std::process;
use std::process::exit;
use std::time::{Instant, SystemTime};
use image::{ImageBuffer, Rgb};
use mnist::{Mnist, MnistBuilder};
use rand::prelude::SliceRandom;
use rand::rng;
use intelligent_artificial::device::GpuContext;
#[allow(unused_imports)]
use intelligent_artificial::nn::functions::{Activation, ErrorFunc, InitHeNormalFunc, InitHeUniformFunc, Normalisation};
use intelligent_artificial::nn::functions::Activation::{Identity, LeakyReLU, Mish};
#[allow(unused_imports)]
use intelligent_artificial::nn::functions::Normalisation::{BatchNorm, Disabled, LayerNorm, RMSNorm};
#[allow(unused_imports)]
use intelligent_artificial::nn::functions::Optimiser::{Adam, SGD};
use intelligent_artificial::nn::functions::Regularisation;
use intelligent_artificial::nn::functions::Regularisation::{L1Regular, L2Regular};
use intelligent_artificial::nn::InitFunc;
#[allow(unused_imports)]
use intelligent_artificial::nn::network::{load_tensors, save_tensors_with_metadata, DenseBlock};
use intelligent_artificial::nn::scheduler::{ReduceLROnPlateauScheduler, SchedulerMode};
use intelligent_artificial::util::Tensor;

const BATCH_SIZE: usize = 200;
const REGU_CONST: f32 = 0.0001;
const CLAMP: f32 = f32::MAX;
const EPSILON: f32 = 0.00000001;
const TRAIN_ELEMENTS: u32 = 60000;
const TEST_ELEMENTS: usize = 10000;

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

        // Find the maximum value in this specific batch row for numerical stability
        let max_logit = row_slice
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        // Compute exponents for this row and sum them up
        let mut sum_exps = 0.0f32;
        for val in row_slice.iter_mut() {
            *val = (*val - max_logit).exp();
            sum_exps += *val;
        }

        // Divide by the sum to turn exponents into probabilities
        for val in row_slice.iter_mut() {
            *val /= sum_exps;
        }
    }

    logits
}

fn main() {
    println!("--- Loading MNIST Dataset ---");
    let mnist = MnistBuilder::new()
        .use_fashion_data()
        .label_format_one_hot()
        .training_set_length(TRAIN_ELEMENTS)
        .test_set_length(TEST_ELEMENTS as u32)
        .download_and_extract()
        .finalize();

    let context = GpuContext::new(16);
    let mut rand = InitHeNormalFunc::new(108, 0.1);

    let mut layers = vec![
        DenseBlock::default(&context, 784, 2048, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 2048, 2048, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 2048, 10, BATCH_SIZE, &mut rand)
    ];

    let mut layers = load_tensors(&context, "data/mnist7AEC.safetensors", BATCH_SIZE);
    // 784 -> 512, 512 -> 256, 256 -> 256, 256 -> 256, 256 -> 512
    while layers.len() > 2 {
        layers.pop();
    }
    layers.push(DenseBlock::default(&context, 256, 10, BATCH_SIZE, &mut rand));

    //configure_layers(&mut layers);
    configure_layers_auto_encoder(&mut layers);

    let mut scheduler = ReduceLROnPlateauScheduler::new(
        8, SchedulerMode::Maximize, 0.05, 0.000001
    );
    scheduler.reset(0.0005);

    for n in 1..=500 {
        epoch(n, &mnist, &mut layers, &context, &mut scheduler);
    }
}

fn configure_layers(layers: &mut Vec<DenseBlock>) {
    layers[0].set_normalisation(LayerNorm);
    layers[0].set_activation(LeakyReLU(0.01));
    layers[0].set_regularisation(L2Regular(REGU_CONST));
    layers[0].set_mask_coeff(0.1);

    layers[1].set_normalisation(LayerNorm);
    layers[1].set_activation(LeakyReLU(0.01));
    layers[1].set_regularisation(L2Regular(REGU_CONST));
    layers[1].set_mask_coeff(0.3);

    layers[2].set_normalisation(Disabled);
    layers[2].set_activation(Identity);
    layers[2].set_regularisation(Regularisation::None);
    layers[2].set_mask_coeff(0.0);
}

fn configure_layers_auto_encoder(layers: &mut Vec<DenseBlock>) {
    // 784 -> 512
    layers[0].set_normalisation(BatchNorm);
    layers[0].set_activation(Mish);

    // 512 -> 256
    layers[1].set_normalisation(BatchNorm);
    layers[1].set_activation(Mish);
}

fn epoch(epoch: usize, mnist: &Mnist, layers: &mut Vec<DenseBlock>, context: &GpuContext, scheduler: &mut ReduceLROnPlateauScheduler) {
    let step_size_offset = (epoch - 1) * TRAIN_ELEMENTS as usize / BATCH_SIZE;

    println!("--- Epoch #{epoch} ---");
    let training_success = train(mnist, layers, context, step_size_offset, scheduler);
    let test_success = test(mnist, layers, context, scheduler);
    println!("--- Training success rate: {training_success}% ---");
    println!("--- Testing success rate: {test_success}% ---");
    println!("--- Epoch #{epoch} complete ---");

    if epoch % 10 == 0 {
        save_tensors_with_metadata(
            &context,
            format!("data/mnist{}.safetensors", epoch/10),
            &vec![&layers[0], &layers[1], &layers[2]],
            &[("a", "b")],
        );
    }
}

fn forward_all<'a>(
    layers: &'a Vec<DenseBlock>, context: &GpuContext,
    input: &'a Tensor, batch_size: usize, is_training: bool, step: usize
) -> [&'a Tensor; 3] {
    let o1 = layers[0].forward(context, input,  batch_size, is_training, step);
    let o2 = layers[1].forward(context, o1,     batch_size, is_training, step);
    let o3 = layers[2].forward(context, o2,     batch_size, is_training, step);
    [o1, o2, o3]
}

fn train(
    mnist: &Mnist, layers: &Vec<DenseBlock>, context: &GpuContext,
    step_offset: usize, scheduler: &mut ReduceLROnPlateauScheduler
) -> f64 {
    let mut indices: Vec<usize> = (0..mnist.trn_img.len() / 784).collect();
    indices.shuffle(&mut rng());

    let adam = Adam(0.9, 0.999, EPSILON);
    let total_batches = indices.chunks(BATCH_SIZE).len();
    let mut success = 0;
    let mut forw_ms = 0.0f64;
    let mut loss_ms = 0.0f64;
    let mut back_ms = 0.0f64;

    for (batch_idx, batch_ids) in indices.chunks(BATCH_SIZE).enumerate() {
        let step = batch_idx + 1 + step_offset;
        let lr = scheduler.get_learning_rate();
        let (input, target_tensor) = build_batch(context, &mnist.trn_img, &mnist.trn_lbl, batch_ids, BATCH_SIZE, 784, 10);

        let t0 = SystemTime::now();
        let outs = forward_all(layers, context, &input, BATCH_SIZE, true, step);
        forw_ms += t0.elapsed().unwrap().as_secs_f64() * 1000.0;

        let t1 = SystemTime::now();
        layers[2].compute_loss(context, &target_tensor, ErrorFunc::CrossEntropyLoss, Activation::Softmax);
        loss_ms += t1.elapsed().unwrap().as_secs_f64() * 1000.0;

        let t2 = SystemTime::now();
        let g2 = layers[2].backward_output(context, outs[1], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        let g1 = layers[1].backward_hidden(context, g2, layers[2].get_weights(), outs[0], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        layers[0].backward_hidden(context, g1, layers[1].get_weights(), &input, BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        back_ms += t2.elapsed().unwrap().as_secs_f64() * 1000.0;

        let out_vec = outs[2].download(context).v;
        assert_no_nan(&out_vec, &format!("training batch {batch_idx}"));
        success += count_correct(&out_vec, &target_tensor.download(context).v, BATCH_SIZE);
    }

    let accuracy = success as f64 / TRAIN_ELEMENTS as f64 * 100.0;

    let avg = |ms: f64| ms / total_batches as f64;
    println!(
        "Forward: {:.2}ms | Loss: {:.2}ms | Backward: {:.2}ms | Total: {:.2}ms | LR: {}",
        avg(forw_ms), avg(loss_ms), avg(back_ms),
        avg(forw_ms + loss_ms + back_ms),
        scheduler.get_learning_rate()
    );

    accuracy
}

fn test(mnist: &Mnist, layers: &Vec<DenseBlock>, context: &GpuContext, scheduler: &mut ReduceLROnPlateauScheduler) -> f64 {
    // Save one reconstruction per class (0-9) for visual inspection
    let mut success = 0;

    for (batch_idx, batch_ids) in (0..TEST_ELEMENTS).collect::<Vec<_>>().chunks(BATCH_SIZE).enumerate() {
        let (input, labels) = build_batch(context, &mnist.tst_img, &mnist.tst_lbl, batch_ids, BATCH_SIZE, 784, 10);
        let outs = forward_all(layers, context, &input, BATCH_SIZE, false, batch_idx);
        let out_mat = outs[2].download(context);

        assert_no_nan(&out_mat.v, &format!("testing batch {batch_idx}"));
        success += count_correct(&out_mat.v, &labels.download(context).v, BATCH_SIZE);
    }

    let accuracy = success as f64 / TEST_ELEMENTS as f64 * 100.0;
    scheduler.step(accuracy as f32);

    accuracy
}

fn count_correct(predictions: &[f32], targets: &[f32], batch_size: usize) -> usize {
    let mut correct = 0;
    for i in 0..batch_size {
        let expected = targets[10 * i..10 * i + 10].iter().position(|&v| v == 1.0).unwrap();
        let predicted = predictions[10 * i..10 * i + 10]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        if expected == predicted {
            correct += 1;
        }
    }
    correct
}

fn build_batch(
    context: &GpuContext, imgs: &[u8], lbls: &[u8],
    ids: &[usize], batch_size: usize, img_size: usize, label_size: usize,
) -> (Tensor, Tensor) {
    let mut pixels = Vec::with_capacity(ids.len() * img_size);
    let mut labels = Vec::with_capacity(ids.len() * label_size);
    for &id in ids {
        pixels.extend(imgs[id * img_size..(id + 1) * img_size].iter().map(|&p| p as f32 / 255.0));
        labels.extend(lbls[id * label_size..(id + 1) * label_size].iter().map(|&l| l as f32));
    }
    (Tensor::from_cpu_vector(context, &pixels, &vec![batch_size, img_size]),
     Tensor::from_cpu_vector(context, &labels, &vec![batch_size, label_size]))
}

fn assert_no_nan(values: &[f32], context: &str) {
    if values.iter().any(|v| v.is_nan()) {
        eprintln!("NAN DETECTED: {context}");
        exit(1);
    }
}