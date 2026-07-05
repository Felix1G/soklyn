use std::collections::HashSet;
use std::fs;
use std::process::exit;
use std::time::SystemTime;
use cifar_ten::*;
use image::{ImageBuffer, Rgb};
use mnist::Mnist;
use rand::prelude::SliceRandom;
use rand::rng;
use intelligent_artificial::device::GpuContext;
use intelligent_artificial::nn::functions::Activation::{Identity, LeakyReLU, Mish};
use intelligent_artificial::nn::functions::{Activation, ErrorFunc, InitHeNormalFunc, Regularisation};
use intelligent_artificial::nn::functions::Normalisation::{BatchNorm, Disabled, LayerNorm};
use intelligent_artificial::nn::functions::Optimiser::Adam;
use intelligent_artificial::nn::functions::Regularisation::L2Regular;
use intelligent_artificial::nn::InitFunc;
use intelligent_artificial::nn::network::{load_tensors, save_tensors_with_metadata, DenseBlock};
use intelligent_artificial::nn::scheduler::{CosineDecayLR, ExponentialLR, ReduceLROnPlateauScheduler, SchedulerMode};
use intelligent_artificial::util::Tensor;

const EPOCHS: usize = 100;
const BATCH_SIZE: usize = 200;
const CLAMP: f32 = f32::MAX;
const EPSILON: f32 = 0.00000001;
const TRAIN_ELEMENTS: u32 = 60000;
const TEST_ELEMENTS: usize = 10000;
const IMG_SIZE: usize = 3072;

struct CifarData {
    trn_img: Vec<u8>,
    trn_lbl: Vec<u8>,
    tst_img: Vec<u8>,
    tst_lbl: Vec<u8>
}

fn main() {
    println!("--- Loading Cifar10 Dataset ---");
    let cifar_res = Cifar10::default()
        .base_path("data/")
        .download_and_extract(false)
        .encode_one_hot(true)
        .build()
        .unwrap();

    let cifar = CifarData {
        trn_img: cifar_res.0, trn_lbl: cifar_res.1, tst_img: cifar_res.2, tst_lbl: cifar_res.3
    };

    let context = GpuContext::new(16);
    let mut rand = InitHeNormalFunc::new(108, 0.1);

    let mut layers = vec![
        DenseBlock::default(&context, IMG_SIZE, 1024, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 1024, 512, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 512, 256, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 256, 128, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, 128, 10, BATCH_SIZE, &mut rand)
    ];

    configure_layers(&mut layers);

    let steps_per_epoch = TRAIN_ELEMENTS as usize / BATCH_SIZE;
    let mut scheduler = CosineDecayLR::new(
        0.00001, 0.001, steps_per_epoch * (EPOCHS - 1), steps_per_epoch
    );

    let mut best = 0.0;
    for n in 1..=EPOCHS {
        let success = epoch(n, &cifar, &mut layers, &context, &mut scheduler);

        if success > best {
            let old_file_path = format!("data/cifar{:.2}.safetensors", best);
            if fs::exists(&old_file_path).unwrap() {
                fs::remove_file(&old_file_path).unwrap();
            }

            best = success;

            let refs: Vec<&DenseBlock> = layers.iter().collect();
            save_tensors_with_metadata(
                &context,
                format!("data/cifar{:.2}.safetensors", best),
                &refs,
                &[("epoch", n.to_string())],
            );
        }
    }
}

fn configure_layers(layers: &mut Vec<DenseBlock>) {
    layers[0].set_normalisation(BatchNorm);
    layers[0].set_activation(Mish);
    layers[0].set_regularisation(L2Regular(0.0001));
    layers[0].set_mask_coeff(0.4);

    layers[1].set_normalisation(BatchNorm);
    layers[1].set_activation(Mish);
    layers[1].set_regularisation(L2Regular(0.0001));
    layers[1].set_mask_coeff(0.4);

    layers[2].set_normalisation(BatchNorm);
    layers[2].set_activation(Mish);
    layers[2].set_regularisation(L2Regular(0.0001));
    layers[2].set_mask_coeff(0.25);

    layers[3].set_normalisation(BatchNorm);
    layers[3].set_activation(Mish);
    layers[3].set_regularisation(L2Regular(0.0001));
    layers[3].set_mask_coeff(0.2);

    layers[4].set_normalisation(Disabled);
    layers[4].set_activation(Identity);
    layers[4].set_mask_coeff(0.0);
}

fn epoch(epoch: usize, cifar: &CifarData, layers: &mut Vec<DenseBlock>, context: &GpuContext, scheduler: &mut CosineDecayLR) -> f64 {
    let step_offset = (epoch - 1) * TRAIN_ELEMENTS as usize / BATCH_SIZE;

    println!("--- Epoch #{epoch} ---");
    let training_success = train(cifar, layers, context, step_offset, scheduler);
    let test_success = test(cifar, layers, context, scheduler);
    println!("--- Training success rate: {training_success}% ---");
    println!("--- Testing success rate: {test_success}% ---");
    println!("--- Epoch #{epoch} complete ---");

    test_success
}

fn forward_all<'a>(
    layers: &'a Vec<DenseBlock>, context: &GpuContext,
    input: &'a Tensor, batch_size: usize, is_training: bool, step: usize
) -> [&'a Tensor; 5] {
    let o1 = layers[0].forward(context, input,  batch_size, is_training, step);
    let o2 = layers[1].forward(context, o1,     batch_size, is_training, step);
    let o3 = layers[2].forward(context, o2,     batch_size, is_training, step);
    let o4 = layers[3].forward(context, o3,     batch_size, is_training, step);
    let o5 = layers[4].forward(context, o4,     batch_size, is_training, step);
    [o1, o2, o3, o4, o5]
}

fn train(
    cifar: &CifarData, layers: &Vec<DenseBlock>, context: &GpuContext,
    step_offset: usize, scheduler: &mut CosineDecayLR
) -> f64 {
    let mut indices: Vec<usize> = (0..cifar.trn_img.len() / IMG_SIZE).collect();
    indices.shuffle(&mut rng());

    let adam = Adam(0.9, 0.999, EPSILON);
    let total_batches = indices.chunks(BATCH_SIZE).len();
    let mut success = 0;
    let mut forw_ms = 0.0f64;
    let mut loss_ms = 0.0f64;
    let mut back_ms = 0.0f64;

    for (batch_idx, batch_ids) in indices.chunks(BATCH_SIZE).enumerate() {
        let step = batch_idx + 1 + step_offset;
        let lr = scheduler.step();
        let (input, target_tensor) = build_batch(context, &cifar.trn_img, &cifar.trn_lbl, batch_ids, BATCH_SIZE, IMG_SIZE, 10);

        let t0 = SystemTime::now();
        let outs = forward_all(layers, context, &input, BATCH_SIZE, true, step);
        forw_ms += t0.elapsed().unwrap().as_secs_f64() * 1000.0;

        let t1 = SystemTime::now();
        layers[4].compute_loss(context, &target_tensor, ErrorFunc::CrossEntropyLoss, Activation::Softmax);
        loss_ms += t1.elapsed().unwrap().as_secs_f64() * 1000.0;

        let t2 = SystemTime::now();
        let g4 = layers[4].backward_output(context, outs[3], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        let g3 = layers[3].backward_hidden(context, g4, layers[4].get_weights(), outs[2], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        let g2 = layers[2].backward_hidden(context, g3, layers[3].get_weights(), outs[1], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        let g1 = layers[1].backward_hidden(context, g2, layers[2].get_weights(), outs[0], BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        layers[0].backward_hidden(context, g1, layers[1].get_weights(), &input, BATCH_SIZE, &adam, &adam, lr, CLAMP, step);
        back_ms += t2.elapsed().unwrap().as_secs_f64() * 1000.0;

        let out_vec = outs[4].download(context).v;
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

fn test(cifar: &CifarData, layers: &Vec<DenseBlock>, context: &GpuContext, scheduler: &mut CosineDecayLR) -> f64 {
    // Save one reconstruction per class (0-9) for visual inspection
    let mut success = 0;

    for (batch_idx, batch_ids) in (0..TEST_ELEMENTS).collect::<Vec<_>>().chunks(BATCH_SIZE).enumerate() {
        let (input, labels) = build_batch(context, &cifar.tst_img, &cifar.tst_lbl, batch_ids, BATCH_SIZE, IMG_SIZE, 10);
        let outs = forward_all(layers, context, &input, BATCH_SIZE, false, batch_idx);
        let out_mat = outs[4].download(context);

        assert_no_nan(&out_mat.v, &format!("testing batch {batch_idx}"));
        success += count_correct(&out_mat.v, &labels.download(context).v, BATCH_SIZE);
    }

    let accuracy = success as f64 / TEST_ELEMENTS as f64 * 100.0;

    accuracy
}

fn count_correct(predictions: &[f32], targets: &[f32], batch_size: usize) -> usize {
    let mut correct = 0;
    for i in 0..batch_size {
        let expected = targets[10 * i..10 * i + 10]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
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

const CIFAR_MEAN: [f32; 3] = [0.4914, 0.4822, 0.4465];
const CIFAR_STD:  [f32; 3] = [0.2470, 0.2435, 0.2616];
const LABEL_SMOOTHING: f32 = 0.1;

fn build_batch(
    context: &GpuContext, imgs: &[u8], lbls: &[u8],
    ids: &[usize], batch_size: usize, img_size: usize, label_size: usize,
) -> (Tensor, Tensor) {
    let mut pixels = Vec::with_capacity(ids.len() * img_size);
    let mut labels = Vec::with_capacity(ids.len() * label_size);
    for &id in ids {
        let img = &imgs[id * img_size..(id + 1) * img_size];

        // Normalize per channel — each channel is 1024 pixels
        for c in 0..3 {
            for i in 0..1024 {
                let raw = img[c * 1024 + i] as f32 / 255.0;
                pixels.push((raw - CIFAR_MEAN[c]) / CIFAR_STD[c]);
            }
        }

        /*for i in 0..label_size {
            let is_correct = lbls[id * label_size + i] == 1;
            let smooth_val = LABEL_SMOOTHING / label_size as f32;
            let correct_val = 1.0 - LABEL_SMOOTHING + smooth_val;
            labels.push(if is_correct { correct_val } else { smooth_val });
        }*/
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