use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr};
use crate::io::device::GpuContext;
use crate::util::log::Error;
use crate::util::precision::PrecisionType;

#[macro_export]
macro_rules! getter {
    ($name:ident, $field:ident . $method:ident(), $t:ty) => {
        pub fn $name(&self) -> &$t {
            &self.$field.$method()
        }
    };

    ($name:ident, $($field:ident).+, $t:ty) => {
        pub fn $name(&self) -> &$t {
            &self.$($field).+
        }
    };
}

#[macro_export]
macro_rules! getter_unwrap {
    ($name:ident, $field:ident . $method:ident(), $t:ty) => {
        pub fn $name(&self) -> &$t {
            &self.$field.$method().+.as_ref().unwrap()
        }
    };

    ($name:ident, $($field:ident).+, $t:ty) => {
        pub fn $name(&self) -> &$t {
            &self.$($field).+.as_ref().unwrap()
        }
    };
}

#[macro_export]
macro_rules! getter_copy {
    ($name:ident, $field:ident . $method:ident(), $t:ty) => {
        pub fn $name(&self) -> $t {
            self.$field.$method()
        }
    };

    ($name:ident, $($field:ident).+, $t:ty) => {
        pub fn $name(&self) -> $t {
            self.$($field).+
        }
    };
}

#[macro_export]
macro_rules! setter {
    ($name:ident, $($field:tt).+, $t:ty) => {
        pub fn $name(&mut self, val: $t) { self.$($field).+ = val; }
    };
}

static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn generate_unique_tensor_id() -> usize {
    TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[allow(unused)]
pub(crate) fn check_all_equal<T: PartialEq>(vec: &Vec<T>) -> bool {
    let t = &vec[0];
    for x in vec.iter().skip(1) {
        if t != x {
            return false;
        }
    }
    true
}

pub(crate) fn scramble_seed(step: u32, layer_id: u32) -> u32 {
    let mut hash = step.wrapping_mul(1103515245).wrapping_add(layer_id);
    hash = (hash ^ (hash >> 16)).wrapping_mul(1103515245);
    hash ^ (hash >> 15)
}

/// Downloads the data from the live GPU memory buffers into the CPU.
/// 
/// # Arguments
/// * `stream` - CUDA stream, can be obtained from [`GpuContext`].
/// * `c` - Slice of allocated live GPU memory buffers to download.
pub fn download_cuda_slice<T: PrecisionType>(stream: &Arc<CudaStream>, c: &CudaSlice<T>) -> Vec<T> {
    stream.clone_dtoh(c).unwrap()
}

/// A typical matrix of size `(rows, cols)`.
#[derive(Debug)]
#[allow(dead_code)]
pub struct Matrix<T: PrecisionType> {
    pub rows: usize,
    pub cols: usize,
    pub v: Vec<T> //values
}

#[allow(dead_code)]
impl<T: PrecisionType> Matrix<T> {
    /// Creates a new empty [`Matrix`]. In other words, size is (0, 0).
    pub fn empty() -> Self { Self::new(0, 0) }

    /// Creates a new [`Matrix`].
    ///
    /// # Arguments
    /// * `rows` - Number of rows.
    ///
    /// * `cols` - Number of columns.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            v: vec![T::zero(); rows * cols]
        }
    }

    /// Creates a new [`Matrix`].
    ///
    /// # Arguments
    /// * `rows` - Number of rows.
    ///
    /// * `cols` - Number of columns.
    ///
    /// * `init` - Initialisation function `which must return a `f32`.
    pub fn new_init<F>(rows: usize, cols: usize, mut init: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut v = Vec::with_capacity(rows * cols);

        for r in 0..rows {
            for c in 0..cols {
                v.push(init(r, c));
            }
        }

        Self { rows, cols, v }
    }

    /// # Arguments
    /// * `row` - The row of the element.
    /// * `col` - The column of the element.
    ///
    /// # Returns
    /// The reference of the element.
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            self.v.get(row * self.cols + col)
        } else {
            None
        }
    }

    /// # Arguments
    /// * `row` - The row of the element.
    /// * `col` - The column of the element.
    ///
    /// # Returns
    /// The mutable reference of the element.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            self.v.get_mut(row * self.cols + col)
        } else {
            None
        }
    }

    /// See [`Self::set_v`].
    /// # Arguments
    /// * `row` - The row of the element.
    /// * `col` - The column of the element.
    pub fn set(&mut self, row: usize, col: usize, val: T) {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.v[row * self.cols + col] = val;
    }

    /// See [`Self::set`].
    /// # Arguments
    /// * `idx` - The index of the value array.
    pub fn set_v(&mut self, idx: usize, val: T) {
        self.v[idx] = val;
    }

    /// Checks if the matrix is empty, which means its size is (0, 0).
    pub fn is_empty(&self) -> bool { self.v.len() == 0 }

    /// Checks if the matrix is not empty, which means its size is not (0, 0).
    pub fn is_not_empty(&self) -> bool { self.v.len() != 0 }
}

/// Multidimensional data container which generalises scalars, vectors, and matrices to
/// any number of dimensions—that serves as the fundamental memory and
/// mathematical tracking unit of a neural network.
/// 
/// Stores some variables in the live GPU memory buffers as `CudaSlice<f32>`. Data is generally row-major.
#[derive(Debug)]
pub struct Tensor<T: PrecisionType> {
    id: usize,
    shape: [usize; 2],
    data: CudaSlice<T>,
}

impl<T: PrecisionType + DeviceRepr> Tensor<T> {
    /// Creates a new tensor from CPU data vector.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `cpu_data` - The data value of the tensor.
    /// * `shape` - The shape of the matrix of this tensor.
    ///
    /// # Panics
    /// Panics if the length of `shape` is not 2, which represents rows and column respectively.
    /// Panics if length of `cpu_data` does not match total elements of the matrix of size `shape`.
    pub fn from_cpu_vector(context: &GpuContext, cpu_data: &[T], shape: &[usize; 2]) -> Self {
        assert_eq!(shape.len(), 2, "Shape must be length of 2 representing rows and columns respectively.");

        let size = shape[0] * shape[1];
        assert_eq!(size, cpu_data.len(), "Length of CPU data does not match shape given in tensor.");

        let data_gpu = context.get_stream().clone_htod(cpu_data).expect("Cannot copy CPU data into VRAM for this tensor.");
        context.get_stream().synchronize().unwrap();

        Self {
            id: generate_unique_tensor_id(),
            shape: shape.clone(),
            data: data_gpu
        }
    }

    #[allow(unused)]
    pub(crate) fn from_gpu_slice(gpu_slice: CudaSlice<T>, shape: &[usize; 2]) -> Self {
        assert_eq!(shape.len(), 2, "Shape must be length of 2 representing rows and columns respectively.");
        assert_eq!(shape.len(), gpu_slice.len(), "Shape and CUDA slice length mismatch.");

        Self {
            id: generate_unique_tensor_id(),
            shape: shape.clone(),
            data: gpu_slice
        }
    }

    /// Creates a new tensor with all the same values.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `shape` - The shape of the matrix of this tensor.
    /// * `value` - The value to be broadcasted to the entire tensor.
    ///
    /// # Panics
    /// Panics if the length of `shape` is not 2, which represents rows and column respectively.
    pub fn fill(context: &GpuContext, shape: &[usize; 2], value: T) -> Self {
        assert_eq!(shape.len(), 2, "Shape must be length of 2 representing rows and columns respectively.");

        let mut tensor = Self::zeros(context, shape);
        tensor.broadcast(context, value);
        tensor
    }

    /// Creates a new empty tensor.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `shape` - The shape of the matrix of this tensor.
    ///
    /// # Panics
    /// Panics if the length of `shape` is not 2, which represents rows and column respectively.
    pub fn zeros(context: &GpuContext, shape: &[usize; 2]) -> Self {
        assert_eq!(shape.len(), 2, "Shape must be length of 2 representing rows and columns respectively.");

        let size = shape[0] * shape[1];
        let data_gpu = context.get_stream().alloc_zeros::<T>(size).expect("Cannot allocate memory in VRAM for this tensor.");
        context.get_stream().synchronize().unwrap();

        Self {
            id: generate_unique_tensor_id(),
            shape: shape.clone(),
            data: data_gpu
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    /// Each tensor possesses its own unique ID.
    pub fn get_id(&self) -> usize { 
        self.id 
    }

    pub fn get_data(&self) -> &CudaSlice<T> {
        &self.data
    }

    /// Safely remove this tensor from memory.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    pub fn drop(self, context: &GpuContext) -> Result<(), Error> {
        drop(self);
        context.get_stream().synchronize()?;
        Ok(())
    }

    /// Fetches `data` from the live GPU memory buffers.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    pub fn download(&self, context: &GpuContext) -> Matrix<T> {
        let mut mat = Matrix::new(self.rows(), self.cols());
        mat.v = download_cuda_slice(context.get_stream(), &self.data);
        mat
    }

    /// Sets all the elements of this tensor to a value in the live GPU memory buffers.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `value` - The value to set.
    pub fn broadcast(&mut self, context: &GpuContext, value: T) {
        context.gpu_broadcast(&self.data, value);
    }

    /// Copies this tensor by copying the data from the live GPU memory buffers.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    pub fn clone(&self, context: &GpuContext) -> Tensor<T> {
        Tensor::<T> {
            id: generate_unique_tensor_id(),
            shape: self.shape.clone(),
            data: context.get_stream().clone_dtod(self.get_data()).expect("Tensor clone failed.")
        }
    }
}