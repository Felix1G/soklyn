use mnist::{Mnist, MnistBuilder};
use rand::prelude::SliceRandom;
use rand::rng;
use soklyn::ffn::FeedForwardNetwork;
use soklyn::io::device::GpuContext;
use soklyn::mlp::DenseBlock;
use soklyn::util::core::Tensor2D;
use soklyn::util::function::Activation::{Identity, LeakyReLU, Mish};
use soklyn::util::function::Normalisation::{BatchNorm, Disabled, LayerNorm};
use soklyn::util::function::Optimiser::Adam;
use soklyn::util::function::Regularisation::L2Regular;
use soklyn::util::function::{Activation, InitFunc, InitHeNormalFunc, LossFunc, Regularisation};
use soklyn::util::log::{init_log, Error};
use soklyn::util::scheduler::{ReduceLROnPlateauScheduler, SchedulerMode};
use std::process::exit;
use std::time::SystemTime;
use soklyn::io::save::SafetensorFile;

const BATCH_SIZE: usize = 200;
const REGU_COEFF: f32 = 0.0001;
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

    for b in 0..batch_dim {
        let row_start = b * num_classes;
        let row_end = row_start + num_classes;
        let row_slice = &mut logits[row_start..row_end];

        let max_logit = row_slice
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exps = 0.0f32;
        for val in row_slice.iter_mut() {
            *val = (*val - max_logit).exp();
            sum_exps += *val;
        }

        for val in row_slice.iter_mut() {
            *val /= sum_exps;
        }
    }

    logits
}

fn main() {
    if let Err(err) = run_pipeline() {
        eprintln!("\n[!] Execution Error: {err}");
        exit(1);
    }
}

fn run_pipeline() -> Result<(), Error> {
    init_log();

    println!("--- Loading MNIST Dataset ---");
    let mnist = MnistBuilder::new()
        .base_path("assets/data/fashion/")
        .use_fashion_data()
        .label_format_one_hot()
        .training_set_length(TRAIN_ELEMENTS)
        .test_set_length(TEST_ELEMENTS as u32)
        .download_and_extract()
        .finalize();

    let context = GpuContext::new(16);
    let mut rand = InitHeNormalFunc::new::<f32>(108, 0.1);

    // --- DEAD CODE RETAINED FOR TESTS ---
    let layers: Vec<DenseBlock<f32>> = vec![
        DenseBlock::default(&context, true, 784, 2048, BATCH_SIZE, &mut rand)?,
        DenseBlock::default(&context, true, 2048, 2048, BATCH_SIZE, &mut rand)?,
        DenseBlock::default(&context, true, 2048, 10, BATCH_SIZE, &mut rand)?
    ];
    
    let writer = SafetensorFile::read("assets/data/mnist7AEC.safetensors")?;
    let mut layers = writer.into_blocks(&context, true, BATCH_SIZE)?;
    while layers.len() > 2 {
        layers.pop();
    }
    layers.push(DenseBlock::default(&context, true, 256, 10, BATCH_SIZE, &mut rand)?);

    configure_layers_auto_encoder(&mut layers);

    let mut network = FeedForwardNetwork::new(layers, 784);

    let mut scheduler = ReduceLROnPlateauScheduler::new(
        8, SchedulerMode::Maximize, 0.05, 0.000001
    );
    scheduler.reset(0.0005);

    for n in 1..=500 {
        epoch(n, &mnist, &mut network, &context, &mut scheduler)?;
    }

    Ok(())
}

fn configure_layers(layers: &mut Vec<DenseBlock<f32>>) {
    layers[0].set_normalisation(LayerNorm);
    layers[0].set_activation(LeakyReLU { coeff: 0.01 });
    layers[0].set_regularisation(L2Regular { regu_coeff: REGU_COEFF });
    layers[0].set_mask_coeff(0.1);

    layers[1].set_normalisation(LayerNorm);
    layers[1].set_activation(LeakyReLU { coeff: 0.01 });
    layers[1].set_regularisation(L2Regular { regu_coeff: REGU_COEFF });
    layers[1].set_mask_coeff(0.3);

    layers[2].set_normalisation(Disabled);
    layers[2].set_activation(Identity);
    layers[2].set_regularisation(Regularisation::None);
    layers[2].set_mask_coeff(0.0);
}

fn configure_layers_auto_encoder(layers: &mut Vec<DenseBlock<f32>>) {
    layers[0].set_normalisation(BatchNorm);
    layers[0].set_activation(Mish);

    layers[1].set_normalisation(BatchNorm);
    layers[1].set_activation(Mish);
}

fn epoch(
    epoch: usize,
    mnist: &Mnist,
    network: &mut FeedForwardNetwork<f32>,
    context: &GpuContext,
    scheduler: &mut ReduceLROnPlateauScheduler,
) -> Result<(), Error> {
    let step_size_offset = (epoch - 1) * TRAIN_ELEMENTS as usize / BATCH_SIZE;

    println!("--- Epoch #{epoch} ---");
    let training_success = train(mnist, network, context, step_size_offset, scheduler)?;
    let test_success = test(mnist, network, context, scheduler)?;
    println!("--- Training success rate: {training_success}% ---");
    println!("--- Testing success rate: {test_success}% ---");
    println!("--- Epoch #{epoch} complete ---");

    if epoch % 10 == 0 {
        let mut writer = SafetensorFile::from_ffn(&context, &network)?;
        writer.pass_metadata(&"epoch", &epoch);
        writer.save(format!("assets/data/mnist{}.safetensors", epoch / 10))?;
    }
    Ok(())
}

fn train(
    mnist: &Mnist,
    network: &mut FeedForwardNetwork<f32>,
    context: &GpuContext,
    step_offset: usize,
    scheduler: &mut ReduceLROnPlateauScheduler,
) -> Result<f64, Error> {
    let mut indices: Vec<usize> = (0..mnist.trn_img.len() / 784).collect();
    indices.shuffle(&mut rng());

    let adam = Adam { m_coeff: 0.9, v_coeff: 0.999, epsilon: EPSILON };
    let total_batches = indices.chunks(BATCH_SIZE).len();
    let mut success = 0;
    let (mut forw_ms, mut loss_ms) = (0.0f64, 0.0f64);

    for (batch_idx, batch_ids) in indices.chunks(BATCH_SIZE).enumerate() {
        let step = batch_idx + 1 + step_offset;
        let lr = scheduler.get_learning_rate();
        let (input, target_tensor) = build_batch(context, &mnist.trn_img, &mnist.trn_lbl, batch_ids, BATCH_SIZE, 784, 10)?;

        let t0 = SystemTime::now();
        let outs = network.forward(context, &input, BATCH_SIZE, true, step)?;
        forw_ms += t0.elapsed().map_err(|_| Error::InvalidConfiguration { reason: "Clock rollback anomaly".to_string() })?.as_secs_f64() * 1000.0;

        let t1 = SystemTime::now();
        network.backward(context, &outs, &target_tensor, &input, LossFunc::CrossEntropyLoss, Activation::Softmax, BATCH_SIZE,
                         &[adam, adam, adam], &[adam, adam, adam], lr, CLAMP, step)?;
        loss_ms += t1.elapsed().map_err(|_| Error::InvalidConfiguration { reason: "Clock rollback anomaly".to_string() })?.as_secs_f64() * 1000.0;

        let out_vec = outs[2].download(context)?.v;
        assert_no_nan(&out_vec, &format!("training batch {batch_idx}"));
        success += count_correct(&out_vec, &target_tensor.download(context)?.v, BATCH_SIZE)?;
    }

    let accuracy = success as f64 / TRAIN_ELEMENTS as f64 * 100.0;
    let avg = |ms: f64| ms / total_batches as f64;

    println!(
        "Forward: {:.2}ms | Backward: {:.2}ms | Total: {:.2}ms | LR: {}",
        avg(forw_ms), avg(loss_ms),
        avg(forw_ms + loss_ms),
        scheduler.get_learning_rate()
    );

    Ok(accuracy)
}

fn test(
    mnist: &Mnist,
    network: &mut FeedForwardNetwork<f32>,
    context: &GpuContext,
    scheduler: &mut ReduceLROnPlateauScheduler,
) -> Result<f64, Error> {
    let mut success = 0;

    for (batch_idx, batch_ids) in (0..TEST_ELEMENTS).collect::<Vec<_>>().chunks(BATCH_SIZE).enumerate() {
        let (input, labels) = build_batch(context, &mnist.tst_img, &mnist.tst_lbl, batch_ids, BATCH_SIZE, 784, 10)?;
        let outs = network.forward(context, &input, BATCH_SIZE,false, batch_idx)?;
        let out_mat = outs[2].download(context)?;

        assert_no_nan(&out_mat.v, &format!("testing batch {batch_idx}"));
        success += count_correct(&out_mat.v, &labels.download(context)?.v, BATCH_SIZE)?;
    }

    let accuracy = success as f64 / TEST_ELEMENTS as f64 * 100.0;
    scheduler.step(accuracy as f32);

    Ok(accuracy)
}

fn count_correct(predictions: &[f32], targets: &[f32], batch_size: usize) -> Result<usize, Error> {
    let mut correct = 0;
    for i in 0..batch_size {
        let expected = targets[10 * i..10 * i + 10].iter().position(|&v| v == 1.0)
            .ok_or_else(|| Error::InvalidConfiguration { reason: "Missing target hot value label pattern".to_string() })?;

        let predicted = predictions[10 * i..10 * i + 10]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| Error::InvalidConfiguration { reason: "Failed mapping logit array slice indices".to_string() })?;

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
) -> Result<(Tensor2D<f32>, Tensor2D<f32>), Error> {
    let mut pixels = Vec::with_capacity(ids.len() * img_size);
    let mut labels = Vec::with_capacity(ids.len() * label_size);

    for &id in ids {
        pixels.extend(imgs[id * img_size..(id + 1) * img_size].iter().map(|&p| p as f32 / 255.0));
        labels.extend(lbls[id * label_size..(id + 1) * label_size].iter().map(|&l| l as f32));
    }

    Ok((
        Tensor2D::from_cpu_vector(context, &pixels, &[batch_size, img_size])?,
        Tensor2D::from_cpu_vector(context, &labels, &[batch_size, label_size])?,
    ))
}

fn assert_no_nan(values: &[f32], context: &str) {
    if values.iter().any(|v| v.is_nan()) {
        eprintln!("NAN DETECTED: {context}");
        exit(1);
    }
}