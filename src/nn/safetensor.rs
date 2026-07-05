use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;
use serde::{Deserialize, Serialize};

pub(crate) struct SafetensorDescriptor {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<f32>
}

#[derive(Serialize, Deserialize, Debug)]
pub (crate) struct JSONTensorDescriptor {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data_offsets: [usize; 2]
}

pub(crate) fn save_safe_tensor<P: AsRef<Path>>(path: P, metadata: Vec<(String, String)>, descriptors: Vec<SafetensorDescriptor>) {
    let mut header_map = serde_json::Map::new();

    // Add Metadata if it exists
    if !metadata.is_empty() {
        header_map.insert("__metadata__".to_string(), serde_json::to_value(&metadata).unwrap());
    }

    // Convert all safetensor descriptors into JSON descriptors
    let mut cur_byte_offset = 0;
    for descriptor in &descriptors {
        let total_elements = descriptor.data.len();
        let start_offset = cur_byte_offset;
        let end_offset = start_offset + (total_elements * 4); // f32 has 4 bytes
        cur_byte_offset = end_offset;

        let desc_val = JSONTensorDescriptor {
            dtype: "F32".to_string(),
            shape: descriptor.shape.clone(),
            data_offsets: [start_offset, end_offset],
        };

        header_map.insert(descriptor.name.clone(), serde_json::to_value(&desc_val).unwrap());
    }

    // Convert to JSON string
    let mut json_buffer = serde_json::to_string_pretty(&header_map).unwrap();

    // Create the .safetensors file
    let mut file = File::create(path).expect("Unable to create safetensors file.");

    // Apply padding of zeros since tensor data memory addresses must be 8-byte aligned relative to the start of the file
    let remainder = json_buffer.len() % 8;
    let padding_needed = if remainder == 0 { 0 } else { 8 - remainder };
    if padding_needed > 0 {
        json_buffer.push_str(&" ".repeat(padding_needed));
    }

    // First 8 bytes are header size in little-endian u64. Then, write the header.
    let buffer_len = json_buffer.len() as u64;
    file.write_all(&buffer_len.to_le_bytes()).expect("Failed to write header size.");
    file.write_all(json_buffer.as_bytes()).expect("Failed to write JSON header.");

    // Write the actual tensor data.
    for descriptor in descriptors {
        let data_bytes: &[u8] = bytemuck::cast_slice(&descriptor.data);
        file.write_all(data_bytes).expect("Failed to write binary tensor data.");
    }

    file.flush().expect("Failed to flush safetensors file.");
}

pub(crate) fn read_safe_tensor<P: AsRef<Path>>(path: P) -> Vec<SafetensorDescriptor> {
    let mut file = File::open(path).expect("Unable to open file.");

    // ================= PARSE HEADER ================
    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes).expect("Unable to read header size.");
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;

    let mut header_bytes = vec![0u8; header_size];
    file.read_exact(&mut header_bytes).expect("Unable to read header bytes.");

    // Deserialize the top-level JSON fields into a flat map
    let header_map: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)
        .expect("Header is not valid JSON.");

    // The binary data starts exactly after the 8-byte prefix + the header size
    let payload_base = 8 + header_size;
    let mut tensor_data = Vec::with_capacity(header_map.len());

    // ================= EXTRACT TENSOR DATA ================
    for (key, val) in header_map {
        if key == "__metadata__" {
            continue; // Skip the global metadata dictionary
        }

        // Safely map the raw JSON value into our specific TensorHeaderItem struct
        let item: JSONTensorDescriptor = match serde_json::from_value(val) {
            Ok(parsed) => parsed,
            Err(e) => {
                panic!("Tensor {key} has an invalid or missing schema structural property: {e}.");
            }
        };

        if item.dtype != "F32" {
            panic!("Tensor {key} has dtype {} which is not supported!", item.dtype);
        }

        // Calculate absolute offsets from the file start
        let start_offset = payload_base + item.data_offsets[0];
        let end_offset = payload_base + item.data_offsets[1];
        let byte_len = end_offset - start_offset;

        // Jump straight to the tensor's binary segment
        file.seek(SeekFrom::Start(start_offset as u64))
            .expect("Failed to seek to tensor data position.");

        let mut tensor_bytes = vec![0u8; byte_len];
        file.read_exact(&mut tensor_bytes)
            .expect("Failed to read binary tensor data bytes.");

        // Fast zero-copy transform: reinterpret the raw byte chunk directly into a vector of f32s.
        let f32_values: Vec<f32> = bytemuck::cast_slice(&tensor_bytes).to_vec();

        tensor_data.push(SafetensorDescriptor {
            name: key,
            shape: item.shape,
            data: f32_values,
        });
    }

    tensor_data
}