<div align="center">

# ⚡ Soklyn (速練)

**A high-performance neural network library written in Rust, utilizing CUDA for GPU-accelerated layer execution.**

[![Language](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)
[![Backend](https://img.shields.io/badge/backend-CUDA-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[<img alt="crates.io" src="https://img.shields.io/crates/v/soklyn.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20">](https://crates.io/crates/soklyn)

[Features](#features) • [Quick Start](#quick-start) • [Architecture](#architecture) • [Usage](#usage)

</div>

---

## Overview

**Soklyn** (derived from **速練** — *Fast Training*) is a minimal, lightweight deep learning abstraction framework designed from the ground up for bare-metal speed.
Built as a personal hobby project, Soklyn was born out of curiosity — what does it actually take to train a neural network from scratch, 
without PyTorch, without TensorFlow, without any abstraction layer between the programmer and the GPU? 
The answer turned out to be: custom CUDA kernels, a lot of debugging, and more debugging until it finally works.
Soklyn is not trying to replace existing frameworks. It exists because building it was worth doing.

You can say that this is a HUGE improvement to my previous project [fksainetwork](https://github.com/Felix1G/fksainetwork) :)

## Features
*This framework currently only supports feed forward networks. Convolutional neural networks and more are coming soon!*

* **Custom CUDA kernels** — forward pass, backward pass, and normalisation all implemented from scratch in CUDA C++
* **Mixed precision training** — FP16 and FP32 support with master weight buffers for numerical stability
* **Multiple normalisations** — RMSNorm, LayerNorm, and BatchNorm
* **Multiple optimisers** — Adam and Stochastic Gradient Descent (SGD) with Nesterov momentum
* **Regularisation** — L1, L2, and Dropout
* **Activations** — ReLU, LeakyReLU, Mish, SiLU, Sigmoid, Tanh, Softmax
* **Loss functions** — Mean Squared Error, Cross Entropy, Binary Cross Entropy
* **Safetensors support** — save and load model weights in the standard safetensors format
* **Multiple learning rate schedulers** — Cosine Decay, Exponential, Reduce LR on Plateau

## Quick Start
Add the following placeholders directly to your local workspace development configurations:

```toml
[dependencies]
soklyn = "0.1.0"
```

## Architecture
Soklyn organizes neural execution state through a streamlined linear spine pattern. 
Instead of a messy global memory architecture, the operations are cleanly divided into different `DenseBlock` layers, 
which can be optionally passed into a `FeedForwardNetwork` wrapper.


The diagram below demonstrates a visualisation of a supposing 3-layer network:
```
[Raw Host Pixels Slice]
           │   (.forward_raw abstraction)
           ▼
 ┌───────────────────┐
 │   Input Tensor    │
 └─────────┬─────────┘
           │
           ▼
 ┌───────────────────┐
 │ DenseBlock Layer0 ├─────────┐
 └─────────┬─────────┘         │
           │                   ▼
           ▼            [Outputs Array]
 ┌───────────────────┐   References  
 │ DenseBlock Layer1 ├─────────┼─────────► [.backward()]
 └─────────┬─────────┘         │
           │                   ▼
           ▼                   │
 ┌───────────────────┐         │
 │ DenseBlock Layer2 ├─────────┘
 └─────────┬─────────┘
           │
           ▼
   [Final Out Tensor]
```

## Usage
```rust
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
                    Normalisation::Disabled, Activation::LeakyReLU(0.01), Regularisation::None, 0.1)?,
    DenseBlock::new(&context, true, 16, 8, BATCH_SIZE, &mut init,
                    Normalisation::Disabled, Activation::LeakyReLU(0.01), Regularisation::None, 0.1)?,
    DenseBlock::new(&context, true, 8, 4, BATCH_SIZE, &mut init,
                    Normalisation::Disabled, Activation::Identity, Regularisation::None, 0.1)?,
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
```
