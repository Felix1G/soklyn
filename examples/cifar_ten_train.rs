use cifar_ten::*;
use half::f16;
use soklyn::io::device::GpuContext;
#[allow(unused_imports)]
use soklyn::layers::DenseBlock;
use soklyn::util::core::Tensor;
use soklyn::util::functions::Activation::{Identity, Mish};
use soklyn::util::functions::Normalisation::{BatchNorm, Disabled};
use soklyn::util::functions::Optimiser::Adam;
use soklyn::util::functions::Regularisation::L2Regular;
use soklyn::util::functions::{Activation, InitHeNormalFunc, InitFunc, LossFunc};
use soklyn::util::log::{init_log, Error};
use soklyn::util::precision::PrecisionType;
use soklyn::util::scheduler::CosineDecayLR;
use rand::prelude::SliceRandom;
use rand::rng;
use std::fs;
use std::process::exit;
use std::time::SystemTime;
use soklyn::io::save::SafetensorFile;

const EPOCHS: usize = 100;
const BATCH_SIZE: usize = 200;
const CLAMP: f32 = f32::MAX;
const EPSILON: f32 = 0.00001;
const TRAIN_ELEMENTS: u32 = 50000;
const TEST_ELEMENTS: usize = 10000;
const IMG_SIZE: usize = 3072;

type Precision = f16;

macro_rules! prec_arr {
    ($($val:expr),* $(,)?) => {
        [$( <Precision as PrecisionType>::from_f32($val as f32) ),*]
    };
}

macro_rules! prec {
    ($val:expr) => {
        <Precision as PrecisionType>::from_f32($val as f32)
    };
}

struct CifarData {
    trn_img: Vec<u8>,
    trn_lbl: Vec<u8>,
    tst_img: Vec<u8>,
    tst_lbl: Vec<u8>,
}

fn main() {
    if let Err(err) = run_pipeline() {
        eprintln!("\n[!] Execution Error: {err}");
        exit(1);
    }
}

fn run_pipeline() -> Result<(), Error> {
    init_log();

    println!("--- Loading Cifar10 Dataset ---");
    let cifar_res = Cifar10::default()
        .base_path("assets/data/")
        .download_and_extract(false)
        .encode_one_hot(true)
        .build()
        .map_err(|e| Error::InvalidConfiguration { reason: format!("Cifar10 loading failed: {e}") })?;

    let cifar = CifarData {
        trn_img: cifar_res.0,
        trn_lbl: cifar_res.1,
        tst_img: cifar_res.2,
        tst_lbl: cifar_res.3,
    };

    let context = GpuContext::new(16);
    let mut rand = InitHeNormalFunc::new::<Precision>(108, 0.3);

    let mut layers: Vec<DenseBlock<Precision>> = vec![
        DenseBlock::default(&context, true, IMG_SIZE, 1024, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, true, 1024, 512, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, true, 512, 256, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, true, 256, 128, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, true, 128, 10, BATCH_SIZE, &mut rand),
        DenseBlock::default(&context, true, 128, 2, BATCH_SIZE, &mut rand)
    ];

    for layer in layers {
        layer.drop(&context)?;
    }

    let mut layers = vec![];
    let writer = SafetensorFile::read("assets/data/cifar53.86.safetensors")?;
    println!("Metadata: {:?}", writer.get_metadata());
    let loaded_blocks = writer.into_blocks(&context, true, BATCH_SIZE)?;

    for layer in loaded_blocks {
        layers.push(layer);
    }
    configure_layers(&mut layers);

    let steps_per_epoch = TRAIN_ELEMENTS as usize / BATCH_SIZE;
    let mut scheduler = CosineDecayLR::new(
        0.00001, 0.001, steps_per_epoch * (EPOCHS - 1), steps_per_epoch
    );

    let mut best = 0.0;
    for n in 1..=EPOCHS {
        let success = epoch(n, &cifar, &mut layers, &context, &mut scheduler)?;

        if success > best {
            let old_file_path = format!("assets/data/cifar{:.2}.safetensors", best);
            if fs::exists(&old_file_path).unwrap_or(false) {
                fs::remove_file(&old_file_path).map_err(Error::IOError)?;
            }

            best = success;

            let mut writer = SafetensorFile::from_blocks(&context, &layers)?;
            writer.pass_metadata(&"epoch", &n);
            writer.save(format!("assets/data/cifar{:.2}.safetensors", best))?;
        }
    }

    Ok(())
}

fn configure_layers(layers: &mut Vec<DenseBlock<Precision>>) {
    let settings = [
        (BatchNorm, Mish, L2Regular(0.0001), 0.4),
        (BatchNorm, Mish, L2Regular(0.0001), 0.4),
        (BatchNorm, Mish, L2Regular(0.0001), 0.25),
        (BatchNorm, Mish, L2Regular(0.0001), 0.2),
        (Disabled, Identity, L2Regular(0.0), 0.0),
        (Disabled, Identity, L2Regular(0.0), 0.0),
    ];

    for (idx, &(norm, act, reg, mask)) in settings.iter().enumerate() {
        layers[idx].set_normalisation(norm);
        layers[idx].set_activation(act);
        if idx < 4 {
            layers[idx].set_regularisation(reg);
        }
        layers[idx].set_mask_coeff(mask);
    }
}

fn epoch(
    epoch: usize,
    cifar: &CifarData,
    layers: &mut Vec<DenseBlock<Precision>>,
    context: &GpuContext,
    scheduler: &mut CosineDecayLR,
) -> Result<f64, Error> {
    let step_offset = (epoch - 1) * TRAIN_ELEMENTS as usize / BATCH_SIZE;

    println!("--- Epoch #{epoch} ---");
    let (training_success1, training_success2) = train(cifar, layers, context, step_offset, scheduler)?;
    let (test_success1, test_success2) = test(cifar, layers, context, scheduler)?;
    println!("--- Training success rate: {training_success1}% | {training_success2}% ---");
    println!("--- Testing success rate: {test_success1}% | {test_success2}% ---");
    println!("--- Epoch #{epoch} complete ---");

    Ok(test_success1)
}

fn forward_all<'a>(
    layers: &'a Vec<DenseBlock<Precision>>,
    context: &GpuContext,
    input: &'a Tensor<Precision>,
    batch_size: usize,
    is_training: bool,
    step: usize,
) -> Result<[&'a Tensor<Precision>; 6], Error> {
    let o0 = layers[0].forward(context, input, batch_size, is_training, step)?;
    let o1 = layers[1].forward(context, o0,    batch_size, is_training, step)?;
    let o2 = layers[2].forward(context, o1,    batch_size, is_training, step)?;
    let o3 = layers[3].forward(context, o2,    batch_size, is_training, step)?;
    let o4 = layers[4].forward(context, o3,    batch_size, is_training, step)?;
    let o5 = layers[5].forward(context, o3,    batch_size, is_training, step)?;
    Ok([o0, o1, o2, o3, o4, o5])
}

fn train(
    cifar: &CifarData,
    layers: &Vec<DenseBlock<Precision>>,
    context: &GpuContext,
    step_offset: usize,
    scheduler: &mut CosineDecayLR,
) -> Result<(f64, f64), Error> {
    let mut indices: Vec<usize> = (0..cifar.trn_img.len() / IMG_SIZE).collect();
    indices.shuffle(&mut rng());

    let adam = Adam(0.9, 0.999, EPSILON);
    let total_batches = indices.chunks(BATCH_SIZE).len();
    let mut success1 = 0;
    let mut success2 = 0;
    let (mut forw_ms, mut loss_ms, mut back_ms) = (0.0f64, 0.0f64, 0.0f64);

    for (batch_idx, batch_ids) in indices.chunks(BATCH_SIZE).enumerate() {
        let (input, target_10, target_2) = build_batch(context, &cifar.trn_img, &cifar.trn_lbl, batch_ids, BATCH_SIZE, IMG_SIZE, 10)?;

        let step = batch_idx + 1 + step_offset;
        let lr = scheduler.step();

        // Alternate every batch: true = 10-class, false = 2-class
        let train_10_class = batch_idx % 20 != 0;

        let t0 = SystemTime::now();
        let outs = forward_all(layers, context, &input, BATCH_SIZE, true, step)?;
        forw_ms += t0.elapsed().map_err(|_| Error::InvalidConfiguration { reason: "Clock went backwards".to_string() })?.as_secs_f64() * 1000.0;

        let t1 = SystemTime::now();
        if train_10_class {
            layers[4].compute_loss(context, &target_10, LossFunc::CrossEntropyLoss, Activation::Softmax)?;
            layers[4].backward_output(context, outs[3], BATCH_SIZE, &adam, &adam, lr, CLAMP, step)?;
            layers[3].backward_hidden(context, &layers[4], outs[2], BATCH_SIZE, &adam, &adam, lr, CLAMP, step)?;
        } else {
            layers[5].compute_loss(context, &target_2, LossFunc::CrossEntropyLoss, Activation::Softmax)?;
            layers[5].backward_output(context, outs[3], BATCH_SIZE, &adam, &adam, lr * 0.01, CLAMP, step)?; // apply your scaling factor safely here!
            layers[3].backward_hidden(context, &layers[5], outs[2], BATCH_SIZE, &adam, &adam, lr * 0.01, CLAMP, step)?;
        }
        loss_ms += t1.elapsed().map_err(|_| Error::InvalidConfiguration { reason: "Clock went backwards".to_string() })?.as_secs_f64() * 1000.0;

        let t2 = SystemTime::now();
        layers[2].backward_hidden(context, &layers[3], outs[1], BATCH_SIZE, &adam, &adam, lr, CLAMP, step)?;
        layers[1].backward_hidden(context, &layers[2], outs[0], BATCH_SIZE, &adam, &adam, lr, CLAMP, step)?;
        layers[0].backward_hidden(context, &layers[1], &input,   BATCH_SIZE, &adam, &adam, lr, CLAMP, step)?;
        back_ms += t2.elapsed().map_err(|_| Error::InvalidConfiguration { reason: "Clock went backwards".to_string() })?.as_secs_f64() * 1000.0;

        let out_vec1 = outs[4].download(context).v;
        let out_vec2 = outs[5].download(context).v;
            assert_no_nan(&out_vec1, &format!("training batch {batch_idx}"));
        assert_no_nan(&out_vec2, &format!("training batch {batch_idx}"));
        success1 += count_correct(&out_vec1, &target_10.download(context).v, BATCH_SIZE, 10)?;
        success2 += count_correct(&out_vec2, &target_2.download(context).v, BATCH_SIZE, 2)?;
    }

    let avg = |ms: f64| ms / total_batches as f64;

    println!(
        "Forward: {:.2}ms | Loss: {:.2}ms | Backward: {:.2}ms | Total: {:.2}ms | LR: {}",
        avg(forw_ms), avg(loss_ms), avg(back_ms),
        avg(forw_ms + loss_ms + back_ms),
        scheduler.get_learning_rate()
    );

    let accuracy1 = success1 as f64 / TRAIN_ELEMENTS as f64 * 100.0;
    let accuracy2 = success2 as f64 / TRAIN_ELEMENTS as f64 * 100.0;
    Ok((accuracy1, accuracy2))
}

fn test(cifar: &CifarData, layers: &Vec<DenseBlock<Precision>>, context: &GpuContext, _scheduler: &mut CosineDecayLR) -> Result<(f64, f64), Error> {
    let mut success1 = 0;
    let mut success2 = 0;

    for (batch_idx, batch_ids) in (0..TEST_ELEMENTS).collect::<Vec<_>>().chunks(BATCH_SIZE).enumerate() {
        let (input, labels1, labels2) = build_batch(context, &cifar.tst_img, &cifar.tst_lbl, batch_ids, BATCH_SIZE, IMG_SIZE, 10)?;
        let outs = forward_all(layers, context, &input, BATCH_SIZE, false, batch_idx)?;
        let out_mat1 = outs[4].download(context);
        let out_mat2 = outs[5].download(context);

        assert_no_nan(&out_mat1.v, &format!("testing batch {batch_idx}"));
        assert_no_nan(&out_mat2.v, &format!("testing batch {batch_idx}"));
        success1 += count_correct(&out_mat1.v, &labels1.download(context).v, BATCH_SIZE, 10)?;
        success2 += count_correct(&out_mat2.v, &labels2.download(context).v, BATCH_SIZE, 2)?;
    }

    Ok((success1 as f64 / TEST_ELEMENTS as f64 * 100.0, success2 as f64 / TEST_ELEMENTS as f64 * 100.0))
}

fn count_correct(predictions: &[Precision], targets: &[Precision], batch_size: usize, size: usize) -> Result<usize, Error> {
    let mut correct = 0;
    for i in 0..batch_size {
        let expected = targets[size * i..size * i + size]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| Error::InvalidConfiguration { reason: "Empty targets block slice encountered".to_string() })?;

        let predicted = predictions[size * i..size * i + size]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| Error::InvalidConfiguration { reason: "Empty predictions block slice encountered".to_string() })?;

        if expected == predicted {
            correct += 1;
        }
    }
    Ok(correct)
}

fn build_batch(
    context: &GpuContext,
    imgs: &[u8],
    lbls: &[u8],
    ids: &[usize],
    batch_size: usize,
    img_size: usize,
    label_size: usize,
) -> Result<(Tensor<Precision>, Tensor<Precision>, Tensor<Precision>), Error> {
    let cifar_mean: [Precision; 3] = prec_arr![0.4914, 0.4822, 0.4465];
    let cifar_std:  [Precision; 3] = prec_arr![0.2470, 0.2435, 0.2616];

    let mut pixels = Vec::with_capacity(ids.len() * img_size);
    let mut labels1 = Vec::with_capacity(ids.len() * label_size);
    let mut labels2 = Vec::with_capacity(2 * label_size);

    for &id in ids {
        let img = &imgs[id * img_size..(id + 1) * img_size];

        // Normalize per channel — each channel is 1024 pixels
        for c in 0..3 {
            for i in 0..1024 {
                let raw = prec!((img[c * 1024 + i] as f32 / 255.0) as f64);
                pixels.push((raw - cifar_mean[c]) / cifar_std[c]);
            }
        }

        labels1.extend(lbls[id * label_size..(id + 1) * label_size].iter().map(|&l| Precision::from(l)));

        let class_idx = lbls.iter().position(|&l| l == 1).unwrap();
        // Animals (Bird=2, Cat=3, Deer=4, Dog=5, Frog=6, Horse=7)
        // Vehicles (Aeroplanes=0, Auto=1, Ship=8, Truck=9)
        let is_vehicle = matches!(class_idx, 0 | 1 | 8 | 9);

        if is_vehicle {
            labels2.extend([prec!(0.0), prec!(1.0)]); // Class 1: Machine
        } else {
            labels2.extend([prec!(1.0), prec!(0.0)]); // Class 0: Living
        }
    }

    Ok((
        Tensor::from_cpu_vector(context, &pixels, &[batch_size, img_size]),
        Tensor::from_cpu_vector(context, &labels1, &[batch_size, label_size]),
        Tensor::from_cpu_vector(context, &labels2, &[batch_size, 2]),
    ))
}

fn assert_no_nan(values: &[Precision], context: &str) {
    if values.iter().any(|v| v.is_nan()) {
        eprintln!("NAN DETECTED: {context}");
        exit(1);
    }
}