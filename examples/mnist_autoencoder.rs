use image::{ImageBuffer, Rgb};
use mnist::{Mnist, MnistBuilder};
use rand::prelude::SliceRandom;
use rand::rng;
use soklyn::ffn::FeedForwardNetwork;
use soklyn::io::device::GpuContext;
use soklyn::mlp::DenseBlock;
use soklyn::util::core::Tensor2D;
use soklyn::util::function::Activation::{Identity, Mish};
use soklyn::util::function::Normalisation::{BatchNorm, Disabled, LayerNorm};
use soklyn::util::function::Optimiser::Adam;
use soklyn::util::function::{Activation, InitFunc, InitXavierNormalFunc, LossFunc};
use soklyn::util::log::{init_log, Error};
use soklyn::util::scheduler::ExponentialLR;
use std::collections::HashSet;
use std::fs;
use std::process::exit;
use std::time::SystemTime;
use soklyn::io::save::SafetensorFile;

const BATCH_SIZE: usize = 200;
const CLAMP: f32 = f32::MAX;
const EPSILON: f32 = 1e-8;
const TRAIN_ELEMENTS: u32 = 60_000;
const TEST_ELEMENTS: usize = 10_000;

fn main() {
    if let Err(err) = run_pipeline() {
        eprintln!("\n[!] Execution Error: {err}");
        exit(1);
    }
}

fn run_pipeline() -> Result<(), Error> {
    init_log();

    println!("--- Loading Fashion-MNIST Dataset ---");
    let mnist = MnistBuilder::new()
        .base_path("assets/data/fashion/")
        .use_fashion_data()
        .label_format_one_hot()
        .training_set_length(TRAIN_ELEMENTS)
        .test_set_length(TEST_ELEMENTS as u32)
        .download_and_extract()
        .finalize();

    let context = GpuContext::new(16);
    let mut rand = InitXavierNormalFunc::new::<f32>(108, 0.5);

    let mut layers = vec![
        DenseBlock::default(&context, true, 784, 512, BATCH_SIZE, &mut rand)?,
        DenseBlock::default(&context, true, 512, 256, BATCH_SIZE, &mut rand)?,
        DenseBlock::default(&context, true, 256, 256, BATCH_SIZE, &mut rand)?,
        DenseBlock::default(&context, true, 256, 256, BATCH_SIZE, &mut rand)?,
        DenseBlock::default(&context, true, 256, 512, BATCH_SIZE, &mut rand)?,
        DenseBlock::default(&context, true, 512, 784, BATCH_SIZE, &mut rand)?,
    ];

    configure_layers(&mut layers);

    let mut network = FeedForwardNetwork::new(layers, 784);

    let mut scheduler = ExponentialLR::new(0.002, 0.95);

    for n in 1..=70 {
        epoch(n, &mnist, &mut network, &context, &mut scheduler)?;
    }

    Ok(())
}

fn configure_layers(layers: &mut Vec<DenseBlock<f32>>) {
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

fn epoch(
    epoch: usize,
    mnist: &Mnist,
    network: &mut FeedForwardNetwork<f32>,
    context: &GpuContext,
    scheduler: &mut ExponentialLR,
) -> Result<(), Error> {
    let step_offset = (epoch - 1) * TRAIN_ELEMENTS as usize / BATCH_SIZE;

    println!("--- Epoch #{epoch} ---");
    train(mnist, network, context, step_offset, scheduler)?;
    test(mnist, network, context)?;
    println!("--- Epoch #{epoch} complete ---");

    if epoch % 10 == 0 {
        let mut writer = SafetensorFile::from_ffn(&context, &network)?;
        writer.pass_metadata(&"epoch", &epoch);
        writer.save(format!("assets/data/mnist{}AEC.safetensors", epoch / 10))?;
    }
    Ok(())
}

fn train(
    mnist: &Mnist,
    network: &mut FeedForwardNetwork<f32>,
    context: &GpuContext,
    step_offset: usize,
    scheduler: &mut ExponentialLR,
) -> Result<(), Error> {
    let mut indices: Vec<usize> = (0..mnist.trn_img.len() / 784).collect();
    indices.shuffle(&mut rng());

    let adam = Adam(0.9, 0.999, EPSILON);
    let total_batches = indices.chunks(BATCH_SIZE).len();
    let (mut forw_ms, mut loss_ms) = (0.0f64, 0.0f64);

    for (batch_idx, batch_ids) in indices.chunks(BATCH_SIZE).enumerate() {
        let step = batch_idx + 1 + step_offset;
        let lr = scheduler.get_learning_rate();
        let (input, _) = build_batch(context, &mnist.trn_img, &mnist.trn_lbl, batch_ids, BATCH_SIZE, 784, 10)?;

        let t0 = SystemTime::now();
        let outs = network.forward(context, &input, BATCH_SIZE, true, step)?;
        forw_ms += t0.elapsed().map_err(|_| Error::InvalidConfiguration { reason: "Clock fault encountered".to_string() })?.as_secs_f64() * 1000.0;

        let t1 = SystemTime::now();
        network.backward(context, &outs, &input, &input, LossFunc::BinaryCrossEntropy, Activation::Sigmoid, BATCH_SIZE,
                         &[adam, adam, adam, adam, adam, adam],
                         &[adam, adam, adam, adam, adam, adam],
                         lr, CLAMP, step)?;
        loss_ms += t1.elapsed().map_err(|_| Error::InvalidConfiguration { reason: "Clock fault encountered".to_string() })?.as_secs_f64() * 1000.0;

        assert_no_nan(&outs[5].download(context)?.v, &format!("training batch {batch_idx}"));
    }

    let avg = |ms: f64| ms / total_batches as f64;
    println!(
        "Forward: {:.2}ms | Backward: {:.2}ms | Total: {:.2}ms | LR: {}",
        avg(forw_ms), avg(loss_ms),
        avg(forw_ms + loss_ms),
        scheduler.get_learning_rate()
    );

    scheduler.step();
    Ok(())
}

fn test(mnist: &Mnist, network: &mut FeedForwardNetwork<f32>, context: &GpuContext) -> Result<(), Error> {
    let mut unseen_classes: HashSet<usize> = (0..10).collect();

    for (batch_idx, batch_ids) in (0..TEST_ELEMENTS).collect::<Vec<_>>().chunks(BATCH_SIZE).enumerate() {
        let (input, labels) = build_batch(context, &mnist.tst_img, &mnist.tst_lbl, batch_ids, BATCH_SIZE, 784, 10)?;
        let outs = network.forward(context, &input, BATCH_SIZE, false, batch_idx)?;
        let out_mat = outs[5].download(context)?;

        assert_no_nan(&out_mat.v, &format!("testing batch {batch_idx}"));

        if !unseen_classes.is_empty() {
            let class = labels[0..10].iter().position(|&v| v == 1.0)
                .ok_or_else(|| Error::InvalidConfiguration { reason: "Invalid target classification matrix".to_string() })?;

            if unseen_classes.remove(&class) {
                save_reconstruction(&out_mat.v, &input.download(context)?.v, class)?;
            }
        }
    }
    Ok(())
}

fn save_reconstruction(output: &[f32], input: &[f32], class: usize) -> Result<(), Error> {
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let to_pixel = |v: f32| (v * 255.5) as u8;

    // Ensure assets directory target output tree exists
    fs::create_dir_all("assets/data/").map_err(Error::IOError)?;

    // Save reconstruction
    let mut img = ImageBuffer::new(28, 28);
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let val = to_pixel(sigmoid(output[y as usize * 28 + x as usize]));
        *pixel = Rgb([val, val, val]);
    }
    img.save(format!("assets/data/{class}_recon_epoch.png"))
        .map_err(|e| Error::InvalidConfiguration { reason: format!("Image write error: {e}") })?;

    // Save original
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let val = to_pixel(input[y as usize * 28 + x as usize]);
        *pixel = Rgb([val, val, val]);
    }
    img.save(format!("assets/data/{class}_real.png"))
        .map_err(|e| Error::InvalidConfiguration { reason: format!("Image write error: {e}") })?;

    Ok(())
}

fn build_batch(
    context: &GpuContext,
    imgs: &[u8],
    lbls: &[u8],
    ids: &[usize],
    batch_size: usize,
    img_size: usize,
    label_size: usize,
) -> Result<(Tensor2D<f32>, Vec<f32>), Error> {
    let mut pixels = Vec::with_capacity(ids.len() * img_size);
    let mut labels = Vec::with_capacity(ids.len() * label_size);

    for &id in ids {
        pixels.extend(imgs[id * img_size..(id + 1) * img_size].iter().map(|&p| p as f32 / 255.0));
        labels.extend(lbls[id * label_size..(id + 1) * label_size].iter().map(|&l| l as f32));
    }

    Ok((
        Tensor2D::from_cpu_vector(context, &pixels, &[batch_size, img_size])?,
        labels,
    ))
}

fn assert_no_nan(values: &[f32], context: &str) {
    if values.iter().any(|v| v.is_nan()) {
        eprintln!("NAN DETECTED: {context}");
        exit(1);
    }
}