use crate::getter;
use crate::util::log::Error;
use crate::util::r#type::Precision;
use indexmap::{indexmap, IndexMap};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom, Write};

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

pub(crate) struct SafetensorWriter {
    descriptors: Vec<SafetensorDescriptor>,
    metadata: IndexMap<String, String>,
}

impl SafetensorWriter {
    pub fn empty() -> Self {
        Self {
            descriptors: vec![],
            metadata: indexmap![],
        }
    }

    pub fn pass(&mut self, name: String, shape: Vec<usize>, data: Vec<u8>, precision: Precision) {
        self.pass_descriptor(SafetensorDescriptor {
            name,
            shape,
            data,
            precision,
        });
    }

    pub fn pass_descriptor(&mut self, desc: SafetensorDescriptor) {
        self.descriptors.push(desc);
    }

    pub fn pass_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    getter!(pub get_descriptors, descriptors, Vec<SafetensorDescriptor>);
    getter!(pub get_metadata, metadata, IndexMap<String, String>);

    /// Reads and parses a Safetensors file from a reader, extracting tensor descriptors
    /// and metadata.
    ///
    /// This function parses the 8-byte little-endian header length, deserializes the JSON
    /// header, extracts optional `__metadata__` attributes, and reads the raw binary tensor
    /// payload into memory.
    ///
    /// # Arguments
    /// * `file` - A mutable reference to a reader that implements both [`Read`] and [`Seek`].
    ///
    /// # Errors
    /// Returns an [`Error`] if:
    /// * An I/O error occurs while reading or seeking within `file` (e.g., unexpected EOF).
    /// * The 8-byte header size integer exceeds the address space of the target architecture ([`usize`]).
    /// * The JSON header is malformed or cannot be parsed into valid tensor descriptors.
    /// * A tensor specifies an unsupported data type (only `"F32"` and `"F16"` are supported).
    pub fn read<R: Read + Seek>(file: &mut R) -> Result<Self, Error> {
        // Read 8-byte little-endian header size prefix
        let mut header_size_bytes = [0u8; 8];
        file.read_exact(&mut header_size_bytes)?;
        let header_size = usize::try_from(u64::from_le_bytes(header_size_bytes))?;

        // Read and parse the JSON header
        let mut header_bytes = vec![0u8; header_size];
        file.read_exact(&mut header_bytes)?;
        let header_map: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_bytes)?;

        // Tensor binary data begins immediately after the 8-byte prefix + header
        let payload_base = 8 + header_size;
        let tensor_count = header_map.len() - usize::from(header_map.contains_key("__metadata__"));
        let mut tensors = Vec::with_capacity(tensor_count);
        let mut metadata = IndexMap::<String, String>::new();

        for (key, val) in header_map {
            if key == "__metadata__" {
                if let Some(meta_object) = val.as_object() {
                    for (m_key, m_val) in meta_object {
                        if let Some(m_str) = m_val.as_str() {
                            metadata.insert(m_key.clone(), m_str.to_string());
                        } else {
                            metadata.insert(m_key.clone(), m_val.to_string());
                        }
                    }
                }
            } else {
                let desc: JSONTensorDescriptor = serde_json::from_value(val)?;

                match desc.dtype.as_str() {
                    "F32" | "F16" => {}
                    other => {
                        return Err(Error::UnrecognizedTensorKey {
                            key,
                            dtype: format!("{other:?}"),
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
        }

        Ok(Self {
            descriptors: tensors,
            metadata,
        })
    }

    /// Serializes and writes tensor descriptors and raw binary payload data out to a writer
    /// following the Safetensors specification format.
    ///
    /// The file is structured with an 8-byte little-endian header length prefix, followed by
    /// an 8-byte aligned JSON header (including optional `__metadata__`), followed immediately
    /// by contiguous raw tensor byte buffers.
    ///
    /// # Arguments
    /// * `file` - A mutable reference to a writer that implements [`Write`].
    ///
    /// # Errors
    /// Returns an [`Error`] if:
    /// * Serializing metadata or tensor descriptors to JSON fails.
    /// * An I/O error occurs while writing bytes to or flushing `file`.
    pub fn write<W: Write>(&self, file: &mut W) -> Result<(), Error> {
        #[cfg(target_endian = "big")]
        compile_error!("Safetensors requires little-endian byte formatting.");

        let mut header_map = serde_json::Map::new();

        if !self.metadata.is_empty() {
            header_map.insert(
                "__metadata__".to_string(),
                serde_json::to_value(&self.metadata)?,
            );
        }

        // Convert all safetensor descriptors into JSON descriptors
        let mut cur_byte_offset = 0;
        for desc in &self.descriptors {
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
        file.write_all(&(json_str.len() as u64).to_le_bytes())?;
        file.write_all(json_str.as_bytes())?;
        for desc in &self.descriptors {
            file.write_all(&desc.data)?;
        }

        file.flush()?;

        Ok(())
    }
}
