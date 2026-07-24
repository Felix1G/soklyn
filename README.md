⚠️ **Status: Active Development**. FFN execution is stable/functional (You can import it from the crate itself). CNN layers and FFT optimizations are currently **Work In Progress (WIP)**.
<div align="center">

# ⚡ Soklyn (速練)

**A high-performance neural network library written in Rust, utilising CUDA for GPU-accelerated layer execution.**
<p align="center">
  <!-- Core Specs & Links -->
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/language-Rust-orange.svg" alt="Language"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/backend-CUDA-green.svg" alt="Backend"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <a href="https://crates.io/crates/soklyn"><img src="https://img.shields.io/crates/v/soklyn.svg?style=for-the-badge&color=fc8d62&logo=rust" height="20" alt="crates.io"></a>
</p>

<p align="center">
  <!-- Stats & Activity -->
  <a href="https://crates.io/crates/soklyn"><img src="https://img.shields.io/crates/d/soklyn?style=flat-square&logo=rust&color=orange" alt="Crates.io Downloads"></a>
  <a href="https://github.com/Felix1G/soklyn"><img src="https://img.shields.io/github/stars/Felix1G/soklyn?style=flat-square&logo=github" alt="GitHub Stars"></a>
  <img src="https://img.shields.io/badge/Maintained%3F-yes-brightgreen?style=flat-square" alt="Maintenance">
  <img src="https://img.shields.io/github/last-commit/Felix1G/soklyn?style=flat-square&color=blue" alt="GitHub last commit">
  <img src="https://img.shields.io/github/languages/code-size/Felix1G/soklyn?style=flat-square" alt="Code Size">
  <img src="https://img.shields.io/badge/clippy-linted-brightgreen?style=flat&logo=rust" alt="Clippy Linted" />
</p>

<p align="center">
  <!-- Tech Details -->
  <img src="https://img.shields.io/badge/edition-2021-orange?style=flat-square&logo=rust" alt="Rust Edition">
  <img src="https://img.shields.io/badge/Compute-sm__120-76B900?style=flat-square&logo=nvidia" alt="NVIDIA Compute">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-blue?style=flat-square" alt="Platform">
  <img src="https://img.shields.io/badge/tested%20on-Windows%2011-0078D4?style=flat-square&logo=windows" alt="Tested On">
</p>

<p align="center">
  <!-- Miscellaneous -->
  <img src="https://img.shields.io/badge/Built%20In-2026-purple?style=flat-square" alt="Year">
</p>

<p align="center">
  <a href="https://github.com/Felix1G/soklyn/discussions"><img src="https://img.shields.io/badge/GitHub-Discussions-181717?style=flat-square&logo=github" alt="Discussions"></a>
  <a href="mailto:felixks110@gmail.com"><img src="https://img.shields.io/badge/Email-Contact-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"></a>
   <a href="https://discord.com/users/mynameisntfelix"><img src="https://img.shields.io/badge/Discord-Direct%20Message-5865F2?style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>        
</p>

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

<a id="features"></a>

## Contributing
PRs and issues are welcome! Feel free to check the [issues page](https://github.com/Felix1G/soklyn/issues).

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
