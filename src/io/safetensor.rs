use crate::util::log::Error;
use crate::util::precision::Precision;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

pub(crate) struct SafetensorDescriptor {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data: Vec<u8>,
    pub(crate) precision: Precision,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct JSONTensorDescriptor {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data_offsets: [usize; 2],
}

pub(crate) fn save_safe_tensor<P: AsRef<Path>>(
    path: P,
    metadata: Vec<(String, String)>,
    descriptors: Vec<SafetensorDescriptor>,
) -> Result<(), Error> {
    #[cfg(target_endian = "big")]
    compile_error!("Safetensors requires little-endian byte formatting.");

    let mut header_map = serde_json::Map::new();

    if !metadata.is_empty() {
        header_map.insert("__metadata__".to_string(), serde_json::to_value(&metadata)?);
    }

    // Convert all safetensor descriptors into JSON descriptors
    let mut cur_byte_offset = 0;
    for desc in &descriptors {
        let end_offset = cur_byte_offset + desc.data.len();

        let json_desc = JSONTensorDescriptor {
            dtype: match desc.precision {
                Precision::FP32 => "F32".to_string(),
                Precision::FP16 => "F16".to_string(),
            },
            shape: desc.shape.clone(),
            data_offsets: [cur_byte_offset, end_offset],
        };

        header_map.insert(desc.name.clone(), serde_json::to_value(&json_desc)?);

        cur_byte_offset = end_offset;
    }

    // Pad to 8-byte alignment so tensor data is aligned relative to file start
    let mut json_str = serde_json::to_string_pretty(&header_map)?;
    let remainder = json_str.len() % 8;
    if remainder != 0 {
        json_str.push_str(&" ".repeat(8 - remainder));
    }

    // Write: [8-byte LE header size] [JSON header] [tensor data...]
    let mut file = File::create(path)?;
    file.write_all(&(json_str.len() as u64).to_le_bytes())?;
    file.write_all(json_str.as_bytes())?;
    for desc in descriptors {
        file.write_all(&desc.data)?;
    }

    file.flush()?;

    Ok(())
}

pub(crate) fn read_safe_tensor<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<SafetensorDescriptor>, Error> {
    let mut file = File::open(&path)?;

    // Read 8-byte little-endian header size prefix
    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes)?;
    let header_size = u64::from_le_bytes(header_size_bytes) as usize;

    // Read and parse the JSON header
    let mut header_bytes = vec![0u8; header_size];
    file.read_exact(&mut header_bytes)?;
    let header_map: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)?;

    // Tensor binary data begins immediately after the 8-byte prefix + header
    let payload_base = 8 + header_size;
    let tensor_count = header_map.len() - header_map.contains_key("__metadata__") as usize;
    let mut tensors = Vec::with_capacity(tensor_count);

    for (key, val) in header_map {
        if key == "__metadata__" {
            continue;
        }

        let desc: JSONTensorDescriptor = serde_json::from_value(val)?;

        match desc.dtype.as_str() {
            "F32" | "F16" => {}
            other => {
                return Err(Error::UnrecognizedTensorKey {
                    key,
                    dtype: format!("{:?}", other),
                });
            }
        }

        let start = payload_base + desc.data_offsets[0];
        let byte_len = desc.data_offsets[1] - desc.data_offsets[0];

        file.seek(SeekFrom::Start(start as u64))?;

        let mut data = vec![0u8; byte_len];
        file.read_exact(&mut data)?;

        tensors.push(SafetensorDescriptor {
            name: key,
            shape: desc.shape,
            data,
            precision: match desc.dtype.as_str() {
                "F16" => Precision::FP16,
                _ => Precision::FP32,
            },
        });
    }

    Ok(tensors)
}
