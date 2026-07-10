use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use indexmap::IndexMap;
use tempfile::NamedTempFile;
use crate::ffn::FeedForwardNetwork;
use crate::getter;
use crate::io::device::GpuContext;
use crate::io::safetensor::{SafetensorWriter};
use crate::layers::{DenseBlock, ParamState};
use crate::util::core::{Matrix, Tensor};
use crate::util::functions::Activation::Identity;
use crate::util::functions::Normalisation::Disabled;
use crate::util::functions::Regularisation;
use crate::util::log::Error;
use crate::util::precision::{Precision, PrecisionType};

pub struct SafetensorFile {
    writer: SafetensorWriter
}

impl SafetensorFile {
    /// Creates a new empty [`SafeFileTensor`].
    pub fn new() -> Self {
        Self {
            writer: SafetensorWriter::empty()
        }
    }

    /// Saves the network into a .safetensors file.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `path` - Path of the save file.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::SerializationCasting`] - The underlying tensor raw numerical buffers fail memory alignment
    ///   or size validation thresholds while being transmuted into binary safe arrays via `bytemuck`.
    /// * [`Error::SerdeJSON`] - The provided metadata block cannot be parsed or mapped into valid JSON string parameters.
    /// * [`Error::IOError`] - The system fails to create the file at the specified `path`, hits a storage capacity allocation
    ///   threshold, or encounters an issue flushing the stream to disk.
    pub fn from_ffn<T: PrecisionType>(context: &GpuContext, network: &FeedForwardNetwork<T>) -> Result<Self, Error> {
        Ok(
            Self::from_blocks(context, &network.get_layers())?
        )
    }

    /// Creates a new [`SafeFileTensor`] which processes all the blocks in the list of
    /// [`DenseBlock`] for saving.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `blocks` - The list of blocks to process.
    pub fn from_blocks<T: PrecisionType>(context: &GpuContext, blocks: &[DenseBlock<T>]) -> Result<Self, Error> {
        let mut file_writer = Self {
            writer: SafetensorWriter::empty()
        };

        file_writer.pass_blocks(context, blocks)?;

        Ok(file_writer)
    }

    /// Pass the list of [`DenseBlock`] which will be processed for saving.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    /// * `blocks` - The list of blocks to process.
    pub fn pass_blocks<T: PrecisionType>(&mut self, context: &GpuContext, blocks: &[DenseBlock<T>]) -> Result<(), Error> {
        for (idx, layer) in blocks.iter().enumerate() {
            self.pass_layer(context, layer, idx + 1)?;
        }

        Ok(())
    }

    /// Adds a single `key-value` pair to the metadata collection.
    ///
    /// # Arguments
    /// * `key` - The metadata key.
    /// * `value` - The metadata value associated with the key.
    pub fn pass_metadata<K, V>(&mut self, key: &K, value: &V)
    where
        K: AsRef<str>,
        V: ToString,
    {
        self.writer.pass_metadata(key.as_ref().to_string(), value.to_string());
    }

    getter!(get_metadata, writer.get_metadata(), IndexMap<String, String>);

    /// Processes a dense block layer and prepares its weights for saving.
    ///
    /// # Arguments
    /// * `context` - The GPU context handle used for managing device memory or executions.
    /// * `layer` - A reference to the [`DenseBlock`] containing the weights and biases to process.
    /// * `layer_num` - The 1-indexed identifier of the layer within the model architecture.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::SerializationCasting`] - The underlying tensor raw numerical buffers fail memory alignment
    ///   or size validation thresholds while being transmuted into binary safe arrays via `bytemuck`.
    pub fn pass_layer<T: PrecisionType>(
        &mut self,
        context: &GpuContext,
        layer: &DenseBlock<T>,
        layer_num: usize
    ) -> Result<(), Error> {
        let weights = layer.get_weights();
        let biases = layer.get_biases();
        let norm_w = layer.get_norm_weights();
        let norm_b = layer.get_norm_biases();
        let w_shape = vec![weights.rows(), weights.cols()];
        let b_shape = vec![biases.cols()];
        let nw_shape = vec![norm_w.cols()];
        let nb_shape = vec![norm_b.cols()];

        // T-precision tensors
        let t_params: &[(&str, Vec<usize>, Vec<T>)] = &[
            ("w", w_shape.clone(), weights.download(context).v),
            ("b", b_shape.clone(), biases.download(context).v),
            ("norm_w", nw_shape.clone(), norm_w.download(context).v),
            ("norm_b", nb_shape.clone(), norm_b.download(context).v),
        ];
        for (suffix, shape, data) in t_params {
            let byte_data = bytemuck::try_cast_slice(data)
                .map_err(|e| Error::SerializationCasting(format!("{e:?}")))?
                .to_vec();

            self.writer.pass(
                format!("layer{layer_num}.{suffix}"),
                shape.clone(),
                byte_data,
                T::precision()
            );
        }

        if layer.is_training_mode() {
            // F32 tensor: moments + master weights
            let mut f32_params: Vec<(String, Vec<usize>, Vec<f32>)> = vec![
                (
                    format!("layer{layer_num}.dv_w"),
                    w_shape.clone(),
                    layer.get_dv_weights().download(context).v,
                ),
                (
                    format!("layer{layer_num}.dv_b"),
                    b_shape.clone(),
                    layer.get_dv_biases().download(context).v,
                ),
                (
                    format!("layer{layer_num}.dm_w"),
                    w_shape.clone(),
                    layer.get_dm_weights().download(context).v,
                ),
                (
                    format!("layer{layer_num}.dm_b"),
                    b_shape.clone(),
                    layer.get_dm_biases().download(context).v,
                ),
                (
                    format!("layer{layer_num}.dv_norm_w"),
                    nw_shape.clone(),
                    layer.get_dv_norm_weights().download(context).v,
                ),
                (
                    format!("layer{layer_num}.dv_norm_b"),
                    nb_shape.clone(),
                    layer.get_dv_norm_biases().download(context).v,
                ),
                (
                    format!("layer{layer_num}.dm_norm_w"),
                    nw_shape.clone(),
                    layer.get_dm_norm_weights().download(context).v,
                ),
                (
                    format!("layer{layer_num}.dm_norm_b"),
                    nb_shape.clone(),
                    layer.get_dm_norm_biases().download(context).v,
                ),
            ];

            // Optional master weight tensors (only present for F16 networks)
            for (suffix, tensor_opt) in [
                ("master_w", layer.get_master_weights()),
                ("master_b", layer.get_master_biases()),
                ("master_norm_w", layer.get_master_norm_weights()),
                ("master_norm_b", layer.get_master_norm_biases()),
            ] {
                if let Some(tensor) = tensor_opt {
                    let tensor: &Tensor<f32> = tensor;
                    f32_params.push((
                        format!("layer{layer_num}.{suffix}"),
                        vec![tensor.rows(), tensor.cols()],
                        tensor.download(context).v,
                    ));
                }
            }

            for (name, shape, data) in f32_params {
                self.writer.pass(
                    name,
                    shape,
                    bytemuck::cast_slice(&data).to_vec(),
                    Precision::FP32
                );
            }
        }

        Ok(())
    }

    /// Saves the current safetensors data atomically to the specified file path.
    ///
    /// This method uses a temporary file in the target directory to ensure that the
    /// write operation is atomic. If the program crashes or disk writing fails midway,
    /// the original file remains unmodified and uncorrupted.
    ///
    /// # Arguments
    /// * `path` - The target file path where the safetensors should be written.
    ///
    /// # Errors
    /// * [`Error::SerdeJSON`] - The provided metadata block cannot be parsed or mapped into valid JSON string parameters.
    /// * [`Error::IOError`] - The system fails to create the file at the specified `path`, hits a storage capacity allocation
    ///   threshold, or encounters an issue flushing the stream to disk.
    /// * [`Error::PersistError`] - The final atomic file rename/persistence step fails (e.g., due to permission issues).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Error> {
        let target_path = path.as_ref();
        let target_dir = target_path.parent().unwrap_or(Path::new("."));

        let mut tmp_file = NamedTempFile::new_in(target_dir)?;
        self.writer.write(&mut tmp_file)?;

        tmp_file.persist(target_path).map_err(|e| e.error)?;

        Ok(())
    }

    /// Reads and deserializes a safetensors file from the specified path.
    ///
    /// # Arguments
    /// * `path` - The file path of the safetensors file to load.
    ///
    /// # Errors
    /// * [`Error::IOError`] - The file does not exist or cannot be opened due to permissions.
    /// * [`Error::UnrecognizedTensorKey`] - The tensor specified an unrecognised data type.
    pub fn read<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let mut file = File::open(path)?;

        Ok(Self {
            writer: SafetensorWriter::read(&mut file)?
        })
    }

    /// Consumes the writer and returns a list of [`DenseBlock`] from the data stored in this writer.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `is_training` - If set to false, tensors related only to the backward pass
    /// will not be generated to save memory.
    /// * `max_batch_size` - Maximum size of a batch
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::InvalidTensorName`] - A tensor identifier starts with "layer" but violates structural format rules
    ///   (e.g., missing dot separator, unparseable layer index, or unrecognised parameter type suffix).
    /// * [`Error::PrecisionMatch`] - A parameter marked strictly for FP32 processing (such as Adam momentum arrays)
    ///   is found encoded as FP16.
    /// * [`Error::NoLayersFound`] - The target file contains no valid layer data entries (`max_layer == 0`).
    /// * [`Error::MissingLayer`] - A layer structure index within the calculated sequence range `(1..=max_layer)`
    ///   is entirely missing from the file.
    /// * [`Error::MissingWeights`] / [`Error::MissingBiases`] - A layer block is found, but its foundational weights
    ///   (`w`) or biases (`b`) matrices are missing.
    /// * [`Error::DriverError`] - An asynchronous hardware or allocation failure occurs while transferring the reconstructed
    ///   matrices into live GPU memory buffers.
    /// * Any underlying disk file-reading or foundational SafeTensors parsing fault occurs (`read_safe_tensor`).
    pub fn into_blocks<T: PrecisionType + Default>(
        self,
        context: &GpuContext,
        is_training: bool,
        max_batch_size: usize,
    ) -> Result<Vec<DenseBlock<T>>, Error> {
        #[derive(PartialEq)]
        enum ParamType {
            W,
            B,
            MasterW,
            MasterB,
            NormW,
            NormB,
            MasterNormW,
            MasterNormB,
            DvW,
            DvB,
            DmW,
            DmB,
            DvNormW,
            DvNormB,
            DmNormW,
            DmNormB,
        }

        impl ParamType {
            fn from_str(s: &str) -> Option<Self> {
                match s {
                    "w" => Some(Self::W),
                    "b" => Some(Self::B),
                    "master_w" => Some(Self::MasterW),
                    "master_b" => Some(Self::MasterB),
                    "norm_w" => Some(Self::NormW),
                    "norm_b" => Some(Self::NormB),
                    "master_norm_w" => Some(Self::MasterNormW),
                    "master_norm_b" => Some(Self::MasterNormB),
                    "dv_w" => Some(Self::DvW),
                    "dv_b" => Some(Self::DvB),
                    "dm_w" => Some(Self::DmW),
                    "dm_b" => Some(Self::DmB),
                    "dv_norm_w" => Some(Self::DvNormW),
                    "dv_norm_b" => Some(Self::DvNormB),
                    "dm_norm_w" => Some(Self::DmNormW),
                    "dm_norm_b" => Some(Self::DmNormB),
                    _ => None,
                }
            }

            fn is_f32_only(&self) -> bool {
                matches!(
                self,
                Self::MasterW
                    | Self::MasterB
                    | Self::MasterNormW
                    | Self::MasterNormB
                    | Self::DvW
                    | Self::DvB
                    | Self::DmW
                    | Self::DmB
                    | Self::DvNormW
                    | Self::DvNormB
                    | Self::DmNormW
                    | Self::DmNormB
            )
            }

            fn is_vector(&self) -> bool {
                matches!(
                self,
                Self::B
                    | Self::DvB
                    | Self::DmB
                    | Self::NormW
                    | Self::NormB
                    | Self::MasterNormW
                    | Self::MasterNormB
                    | Self::DvNormW
                    | Self::DvNormB
                    | Self::DmNormW
                    | Self::DmNormB
            )
            }
        }

        #[derive(Default)]
        struct LayerParams<T: PrecisionType> {
            w: Option<Matrix<T>>,
            b: Option<Matrix<T>>,
            master_w: Option<Matrix<f32>>,
            master_b: Option<Matrix<f32>>,
            norm_w: Option<Matrix<T>>,
            norm_b: Option<Matrix<T>>,
            master_norm_w: Option<Matrix<f32>>,
            master_norm_b: Option<Matrix<f32>>,
            dv_w: Option<Matrix<f32>>,
            dv_b: Option<Matrix<f32>>,
            dm_w: Option<Matrix<f32>>,
            dm_b: Option<Matrix<f32>>,
            dv_norm_w: Option<Matrix<f32>>,
            dv_norm_b: Option<Matrix<f32>>,
            dm_norm_w: Option<Matrix<f32>>,
            dm_norm_b: Option<Matrix<f32>>,
        }

        let mut layers_map: HashMap<usize, LayerParams<T>> = HashMap::new();
        let mut max_layer = 0usize;

        for tensor in self.writer.get_descriptors() {
            let name = &tensor.name;

            // Parse "layerN.param_suffix"
            let Some(rest) = name.strip_prefix("layer") else {
                log::debug!("Skipping non-layer key: '{name}'");
                continue;
            };

            let (layer_str, param_str) = rest.split_once('.')
                .ok_or_else(|| Error::InvalidTensorName {
                    name: name.clone(),
                    reason: "missing '.' separator"
                })?;

            let layer_idx = layer_str.parse::<usize>()
                .map_err(|_| Error::InvalidTensorName {
                    name: name.clone(),
                    reason: "invalid layer index number"
                })?;

            let param_type = ParamType::from_str(param_str)
                .ok_or_else(|| Error::InvalidTensorName {
                    name: name.clone(),
                    reason: "unrecognized parameter type"
                })?;

            // F32-only params must not be stored as F16
            if param_type.is_f32_only() && tensor.precision == Precision::FP16 {
                return Err(Error::PrecisionMatch {
                    layer: String::from(layer_str),
                    param: String::from(param_str),
                });
            }

            let (rows, cols) = if param_type.is_vector() {
                (1, tensor.shape[0])
            } else {
                (tensor.shape[0], tensor.shape[1])
            };

            let entry = layers_map.entry(layer_idx).or_default();
            max_layer = max_layer.max(layer_idx);

            if param_type.is_f32_only() {
                let mut mat: Matrix<f32> = Matrix::new(rows, cols);
                mat.v = bytemuck::pod_collect_to_vec(&tensor.data);
                match param_type {
                    ParamType::MasterW => entry.master_w = Some(mat),
                    ParamType::MasterB => entry.master_b = Some(mat),
                    ParamType::MasterNormW => entry.master_norm_w = Some(mat),
                    ParamType::MasterNormB => entry.master_norm_b = Some(mat),
                    _ => {
                        if is_training {
                            match param_type {
                                ParamType::DvW => entry.dv_w = Some(mat),
                                ParamType::DvB => entry.dv_b = Some(mat),
                                ParamType::DmW => entry.dm_w = Some(mat),
                                ParamType::DmB => entry.dm_b = Some(mat),
                                ParamType::DvNormW => entry.dv_norm_w = Some(mat),
                                ParamType::DvNormB => entry.dv_norm_b = Some(mat),
                                ParamType::DmNormW => entry.dm_norm_w = Some(mat),
                                ParamType::DmNormB => entry.dm_norm_b = Some(mat),
                                _ => {}
                            }
                        }
                    }
                }
            } else {
                let mut mat: Matrix<T> = Matrix::new(rows, cols);
                mat.v = bytemuck::pod_collect_to_vec(&tensor.data);
                match param_type {
                    ParamType::W => entry.w = Some(mat),
                    ParamType::B => entry.b = Some(mat),
                    ParamType::NormW => entry.norm_w = Some(mat),
                    ParamType::NormB => entry.norm_b = Some(mat),
                    _ => {}
                }
            }
        }

        if max_layer == 0 {
            return Err(Error::NoLayersFound);
        }

        // Helper closures
        let from_t = |opt: &Option<Matrix<T>>, rows, cols| match opt {
            Some(m) => Tensor::<T>::from_cpu_vector(context, &m.v, &[rows, cols]),
            None => Tensor::<T>::zeros(context, &[rows, cols]),
        };

        let from_f32_opt = |opt: &Option<Matrix<f32>>, rows, cols| {
            opt.as_ref()
                .map(|m| Tensor::<f32>::from_cpu_vector(context, &m.v, &[rows, cols]))
        };

        (1..=max_layer)
            .map(|idx| {
                let layer = layers_map
                    .remove(&idx)
                    .ok_or(Error::MissingLayer { layer: idx })?;
                let w = layer
                    .w
                    .as_ref()
                    .ok_or(Error::MissingWeights { layer: idx })?;
                let b = layer
                    .b
                    .as_ref()
                    .ok_or(Error::MissingBiases { layer: idx })?;

                let (wr, wc) = (w.rows, w.cols);
                let (br, bc) = (b.rows, b.cols);

                DenseBlock::<T>::from_tensors(
                    context,
                    is_training,
                    ParamState::from_tensors(
                        Tensor::<T>::from_cpu_vector(context, &w.v, &[wr, wc]),
                        Tensor::<T>::from_cpu_vector(context, &b.v, &[br, bc]),
                        from_f32_opt(&layer.master_w, wr, wc),
                        from_f32_opt(&layer.master_b, br, bc),
                        from_f32_opt(&layer.dv_w, wr, wc),
                        from_f32_opt(&layer.dv_b, br, bc),
                        from_f32_opt(&layer.dm_w, wr, wc),
                        from_f32_opt(&layer.dm_b, br, bc),
                    ),
                    ParamState::from_tensors(
                        from_t(&layer.norm_w, br, wc),
                        from_t(&layer.norm_b, br, bc),
                        from_f32_opt(&layer.master_norm_w, br, wc),
                        from_f32_opt(&layer.master_norm_b, br, bc),
                        from_f32_opt(&layer.dv_norm_w, br, wc),
                        from_f32_opt(&layer.dv_norm_b, br, bc),
                        from_f32_opt(&layer.dm_norm_w, br, wc),
                        from_f32_opt(&layer.dm_norm_b, br, bc),
                    ),
                    Disabled,
                    Identity,
                    Regularisation::None,
                    max_batch_size,
                    0.0,
                )
            })
            .collect::<Result<Vec<DenseBlock<T>>, Error>>()
    }
}