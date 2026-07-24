#[allow(clippy::similar_names)]
mod conv_device;
#[allow(clippy::similar_names)]
mod ffn_device;
#[allow(unused)]
#[allow(clippy::similar_names)]
mod util_device;

use crate::util::core::Tensor2D;
use crate::util::log::Error;
use crate::util::r#type::{Precision, PrecisionType};
use cudarc::driver::sys::CUctx_flags;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchArgs, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Carries the context of the connection to a GPU device.
#[allow(dead_code)]
pub struct GpuContext {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // Utility kernels
    cast_f32_f16_func: CudaFunction,
    cast_f16_f32_func: CudaFunction,
    broadcast_func: (CudaFunction, CudaFunction),
    gemm_func: (CudaFunction, CudaFunction), // A * B
    geam_func: (CudaFunction, CudaFunction), // A + B

    // FFN kernel series
    forward_pass_func0: (CudaFunction, CudaFunction), // Z = WX + B
    forward_pass_func1: (CudaFunction, CudaFunction), // Normalisation
    forward_pass_func2: (CudaFunction, CudaFunction), // Activation and dropout
    softmax_func: (CudaFunction, CudaFunction),
    compute_output_layer_error_func: (CudaFunction, CudaFunction),
    compute_hidden_layer_error_func: (CudaFunction, CudaFunction),
    backward_pass_func: (CudaFunction, CudaFunction),

    // CNN kernel series
    conv_forward_pass_func0: (CudaFunction, CudaFunction), // CONV Z = WX + B
    conv_forward_pass_func1: (CudaFunction, CudaFunction), // Normalisation
    conv_forward_pass_func2: (CudaFunction, CudaFunction), // Activation, Pooling, Dropout

    // CNN FFT kernel series
    conv_fft_row_transform_func: (CudaFunction, CudaFunction),
    conv_fft_col_transform_func: CudaFunction,
    conv_elem_mul_ifft_row_func: CudaFunction,
    conv_ifft_col_transform_func: (CudaFunction, CudaFunction),

    tile_dim: u32,
    tile_dim_minus_1: u32,
    tile_dim_2: u32,
    tile_dim_2_minus_1: u32,
}

#[allow(dead_code)]
impl GpuContext {
    fn new_from_context(context: Arc<CudaContext>, tile_dim: u32) -> Self {
        let nn_math_path = include_str!(concat!(env!("OUT_DIR"), "/cuda/nn_math.ptx"));
        let nn_math_src = Ptx::from_src(nn_math_path);
        let nn_math_mod = context
            .load_module(nn_math_src)
            .expect("Failed to load module 'nn_math.ptx'");

        let load_kernel = |name: &str| {
            nn_math_mod
                .load_function(name)
                .unwrap_or_else(|e| panic!("Failed to load CUDA function '{name}': {e:?}"))
        };

        let cast_f32_f16 = load_kernel("cast_f32_f16_t_kernel");
        let cast_f16_f32 = load_kernel("cast_f16_f32_t_kernel");
        let broadcast_f32 = load_kernel("broadcast_kernel_f32");
        let gemm_f32 = load_kernel("sgemm_kernel");
        let geam_f32 = load_kernel("sgeam_kernel");
        let forward_pass_0_f32 = load_kernel("forward_pass_0_f32");
        let forward_pass_1_f32 = load_kernel("forward_pass_1_f32");
        let forward_pass_2_f32 = load_kernel("forward_pass_2_f32");
        let softmax_f32 = load_kernel("softmax_kernel_f32");
        let compute_output_layer_error_f32 = load_kernel("compute_output_layer_error_f32");
        let compute_hidden_layer_error_f32 = load_kernel("compute_hidden_layer_error_f32");
        let backward_pass_f32 = load_kernel("backward_pass_f32");
        let broadcast_f16 = load_kernel("broadcast_kernel_f16");
        let gemm_f16 = load_kernel("hgemm_kernel");
        let geam_f16 = load_kernel("hgeam_kernel");
        let forward_pass_0_f16 = load_kernel("forward_pass_0_f16");
        let forward_pass_1_f16 = load_kernel("forward_pass_1_f16");
        let forward_pass_2_f16 = load_kernel("forward_pass_2_f16");
        let softmax_f16 = load_kernel("softmax_kernel_f16");
        let compute_output_layer_error_f16 = load_kernel("compute_output_layer_error_f16");
        let compute_hidden_layer_error_f16 = load_kernel("compute_hidden_layer_error_f16");
        let backward_pass_f16 = load_kernel("backward_pass_f16");
        let conv_forward_pass_0_f32 = load_kernel("conv_forward_pass_0_kernel_f32");
        let conv_forward_pass_0_f16 = load_kernel("conv_forward_pass_0_kernel_f16");
        let conv_forward_pass_1_f32 = load_kernel("conv_forward_pass_1_kernel_f32");
        let conv_forward_pass_1_f16 = load_kernel("conv_forward_pass_1_kernel_f16");
        let conv_forward_pass_2_f32 = load_kernel("conv_forward_pass_2_kernel_f32");
        let conv_forward_pass_2_f16 = load_kernel("conv_forward_pass_2_kernel_f16");
        let conv_fft_row_transform_f32 = load_kernel("conv_fft_row_transform_kernel_f32");
        let conv_fft_row_transform_f16 = load_kernel("conv_fft_row_transform_kernel_f16");
        let conv_fft_col_transform_func = load_kernel("conv_fft_col_transform_kernel");
        let conv_elem_mul_ifft_row_func = load_kernel("conv_elem_mul_ifft_row_kernel");
        let conv_ifft_col_transform_f32 = load_kernel("conv_ifft_col_transform_kernel_f32");
        let conv_ifft_col_transform_f16 = load_kernel("conv_ifft_col_transform_kernel_f16");

        let stream = CudaContext::new_stream(&context).expect("Failed to create stream");
        context
            .set_flags(CUctx_flags::CU_CTX_SCHED_SPIN)
            .expect("Failed to set context flags");

        let tile_dim_2 = tile_dim * tile_dim;

        Self {
            context,
            stream,
            cast_f32_f16_func: cast_f32_f16,
            cast_f16_f32_func: cast_f16_f32,
            broadcast_func: (broadcast_f32, broadcast_f16),
            gemm_func: (gemm_f32, gemm_f16),
            geam_func: (geam_f32, geam_f16),
            forward_pass_func0: (forward_pass_0_f32, forward_pass_0_f16),
            forward_pass_func1: (forward_pass_1_f32, forward_pass_1_f16),
            forward_pass_func2: (forward_pass_2_f32, forward_pass_2_f16),
            softmax_func: (softmax_f32, softmax_f16),
            compute_output_layer_error_func: (
                compute_output_layer_error_f32,
                compute_output_layer_error_f16,
            ),
            compute_hidden_layer_error_func: (
                compute_hidden_layer_error_f32,
                compute_hidden_layer_error_f16,
            ),
            backward_pass_func: (backward_pass_f32, backward_pass_f16),
            conv_forward_pass_func0: (conv_forward_pass_0_f32, conv_forward_pass_0_f16),
            conv_forward_pass_func1: (conv_forward_pass_1_f32, conv_forward_pass_1_f16),
            conv_forward_pass_func2: (conv_forward_pass_2_f32, conv_forward_pass_2_f16),
            conv_fft_row_transform_func: (conv_fft_row_transform_f32, conv_fft_row_transform_f16),
            conv_fft_col_transform_func,
            conv_elem_mul_ifft_row_func,
            conv_ifft_col_transform_func: (
                conv_ifft_col_transform_f32,
                conv_ifft_col_transform_f16,
            ),
            tile_dim,
            tile_dim_minus_1: tile_dim - 1,
            tile_dim_2,
            tile_dim_2_minus_1: tile_dim_2 - 1,
        }
    }

    /// Initialises a connection to the first available NVIDIA graphics card.
    ///
    /// # Arguments
    /// * `tile_dim` - Tile Dimension, the size of each block `(tile_dim, tile_dim)`.
    ///   Recommended values are `16` or `32`.
    ///
    /// # Panics
    /// Panics if no compatible NVIDIA GPU is found, or if the CUDA driver
    /// is missing/incompatible.
    #[must_use]
    pub fn new(tile_dim: u32) -> Self {
        let context = CudaContext::new(0).expect("Failed to create CUDA context");
        Self::new_from_context(context, tile_dim)
    }

    /// Initialises a connection to an available NVIDIA graphics card.
    ///
    /// # Arguments
    /// * `ordinal` - The NVIDIA device index. If you only have one device, set this to `0`.
    /// * `tile_dim` - Tile Dimension, the size of each block `(tile_dim, tile_dim)`.
    ///   Recommended values are `16` or `32`.
    ///
    /// # Panics
    /// Panics if no compatible NVIDIA GPU is found, or if the CUDA driver
    /// is missing/incompatible.
    #[must_use]
    pub fn with_device_index(ordinal: usize, tile_dim: u32) -> Self {
        let context = CudaContext::new(ordinal).expect("Failed to create CUDA context");
        Self::new_from_context(context, tile_dim)
    }

    pub(crate) fn get_context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    pub(crate) fn get_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub(crate) fn get_tile_dim(&self) -> u32 {
        self.tile_dim
    }

    pub(crate) fn get_tile_dim_2(&self) -> u32 {
        self.tile_dim_2
    }

    /// # Errors
    /// Returns an [`Error`] if data cannot be downloaded from the GPU.
    pub fn download<K: DeviceRepr>(&self, cuda_slice: &CudaSlice<K>) -> Result<Vec<K>, Error> {
        Ok(self.stream.clone_dtoh(cuda_slice)?)
    }

    /// Calculates the kernel launch config for 2D grids, mostly used by the FFN.
    ///
    /// # Errors
    /// Returns an [`Error`] if the parameters cannot be cast to usize.
    fn calculate_cfg2d(&self, x: usize, y: usize, mem: u32) -> Result<LaunchConfig, Error> {
        Ok(LaunchConfig {
            grid_dim: (
                (u32::try_from(x)? + self.tile_dim_minus_1) / self.tile_dim,
                (u32::try_from(y)? + self.tile_dim_minus_1) / self.tile_dim,
                1,
            ),
            block_dim: (self.tile_dim, self.tile_dim, 1),
            shared_mem_bytes: mem,
        })
    }

    /// Calculates the kernel launch config for 4D tensor processing using all 3 dimensions of the grid.
    /// The batch (`n`) and channels (`c`) are fused into the z-axis.
    /// Spatial height (`h`) takes the y-axis while spatial width (`w`) takes the x-axis.
    ///
    /// # Errors
    /// Returns an [`Error`] if the parameters cannot be cast to usize.
    fn calculate_cfg4d(
        &self,
        n: usize,
        c: usize,
        h: usize,
        w: usize,
        mem: u32,
    ) -> Result<LaunchConfig, Error> {
        Ok(LaunchConfig {
            grid_dim: (
                (u32::try_from(w)? + self.tile_dim_minus_1) / self.tile_dim,
                (u32::try_from(h)? + self.tile_dim_minus_1) / self.tile_dim,
                u32::try_from(n * c)?,
            ),
            block_dim: (self.tile_dim, self.tile_dim, 1),
            shared_mem_bytes: mem,
        })
    }

    fn check_tile_dim<T: PrecisionType>(&self) -> Result<(), Error> {
        if T::precision() == Precision::FP16 && self.tile_dim != 16 {
            return Err(Error::HardwareConstraintViolation {
                precision: Precision::FP16,
                expected: 16,
                found: self.tile_dim,
            });
        }

        Ok(())
    }

    // Used to switch between master tensors and tensors during training,
    // specifically the weights, biases, norm weights, and norm biases tensor.
    fn master_tensor<'a, T: PrecisionType>(
        builder: &mut LaunchArgs<'a>,
        tensor: Option<&'a Tensor2D<f32>>,
        fallback: &'a Tensor2D<T>,
    ) {
        if let Some(tensor_data) = tensor {
            builder.arg(tensor_data.get_data());
        } else {
            builder.arg(fallback.get_data());
        }
    }
}
