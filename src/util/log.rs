use std::io;
use cudarc::driver::DriverError;
use tempfile::PersistError;
use thiserror::Error;
use crate::util::r#type::Precision;

/// Activate logging which includes debug info and detailed errors.
///
/// This function should be called only once at the beginning of your program.
pub fn init_log() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("off")).init();
}

#[derive(Error, Debug)]
pub enum Error {
    #[error("JSON error: {0}")]
    SerdeJSON(#[from] serde_json::Error),

    #[error("I/O error: {0}")]
    IOError(#[from] io::Error),

    #[error("Driver error: {0}")]
    DriverError(#[from] DriverError),

    #[error("Temp file writing error: {0}")]
    PersistError(#[from] PersistError),

    #[error("Index out of bounds: {reason}: {idx} for length {max_idx}.")]
    IndexOutOfBounds {
        idx: usize,
        max_idx: usize,
        reason: &'static str
    },

    #[error("Hardware acceleration constraint violation: Precision is {precision:?}, which requires a tile dimension of {expected}, but found {found}. (WMMA requirement)")]
    HardwareConstraintViolation {
        precision: Precision,
        expected: usize,
        found: u32
    },

    #[error("Memory alignment or size mismatch during serialization: {1}: {0}.")]
    SerializationCasting(String, &'static str),

    #[error("Layer {layer}: '{param}' must be FP32 but safetensor file contains FP16.")]
    PrecisionMatch {
        layer: String,
        param: String
    },

    #[error("Unsupported precision cast operation: Cannot convert data from {from:?} to {to:?}")]
    UnsupportedTypeCast { from: Precision, to: Precision },

    #[error("No network layers were found in safetensor file.")]
    NoLayersFound,

    #[error("Layer {layer} is completely missing from the safetensor file.")]
    MissingLayer {
        layer: usize
    },

    #[error("Weights for layer {layer} is completely missing from the safetensor file.")]
    MissingWeights {
        layer: usize
    },

    #[error("Biases for layer {layer} is completely missing from the safetensor file.")]
    MissingBiases {
        layer: usize
    },

    #[error("Tensor '{key}' has unrecognised dtype '{dtype}'.")]
    UnrecognizedTensorKey {
        key: String,
        dtype: String
    },

    #[error("Invalid tensor name format '{name}': {reason}.")]
    InvalidTensorName { name: String, reason: &'static str },

    #[error("Invalid network configuration: {reason}")]
    InvalidConfiguration { reason: String },

    #[error("Invalid operation: {reason}: Action requires the network to be in training mode.")]
    TrainingModeRequired { reason: &'static str },

    #[error("Allocation limit exceeded: {reason}: items received ({received}) exceeds maximum allowed allocation ({max}).")]
    AllocationLimitExceeded { received: usize, max: usize, reason: &'static str },

    #[error("Dimension mismatch: {reason}: expected {expected}, found {found}.")]
    MismatchedDimensions {
        reason: &'static str,
        expected: usize,
        found: usize,
    },

    #[error("Batch size constraint violation: {reason}")]
    InvalidBatchSize { reason: &'static str },
}

pub type Result<T> = std::result::Result<T, Error>;