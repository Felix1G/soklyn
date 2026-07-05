use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use cudarc::driver::{CudaSlice, CudaStream};
use crate::device::GpuContext;

static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn generate_unique_tensor_id() -> usize {
    TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

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

/// Downloads the data from the VRAM into the CPU.
/// 
/// # Arguments
/// * `stream` - CUDA stream, can be obtained from [`GpuContext`].
/// * `c` - Slice of allocated VRAM memory to download.
pub fn download_cuda_slice(stream: &Arc<CudaStream>, c: &CudaSlice<f32>) -> Vec<f32> {
    stream.clone_dtoh(c).unwrap()
}

/// A typical matrix of size `(rows, cols)`.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub v: Vec<f32> //values
}

#[allow(dead_code)]
impl Matrix {
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
            v: vec![0.0; rows * cols]
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
        F: FnMut(usize, usize) -> f32,
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
    pub fn get(&self, row: usize, col: usize) -> Option<&f32> {
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
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut f32> {
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
    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.v[row * self.cols + col] = val;
    }

    /// See [`Self::set`].
    /// # Arguments
    /// * `idx` - The index of the value array.
    pub fn set_v(&mut self, idx: usize, val: f32) {
        self.v[idx] = val;
    }

    /// Checks if the matrix is empty, which means its size is (0, 0).
    pub fn is_empty(&self) -> bool { self.v.len() == 0 }

    /// Checks if the matrix is not empty, which means its size is not (0, 0).
    pub fn is_not_empty(&self) -> bool { self.v.len() != 0 }
}

/// Multi-dimensional data container which generalises scalars, vectors, and matrices to
/// any number of dimensions—that serves as the fundamental memory and
/// mathematical tracking unit of a neural network.
/// 
/// Stores some variables in VRAM as `CudaSlice<f32>`. Data is generally row-major.
#[derive(Debug)]
pub struct Tensor {
    id: usize,
    shape: Vec<usize>,
    data: CudaSlice<f32>,
}

impl Tensor {
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
    pub fn from_cpu_vector(context: &GpuContext, cpu_data: &Vec<f32>, shape: &Vec<usize>) -> Self {
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

    /// Creates a new tensor with all the same values.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `shape` - The shape of the matrix of this tensor.
    /// * `value` - The value to be broadcasted to the entire tensor.
    ///
    /// # Panics
    /// Panics if the length of `shape` is not 2, which represents rows and column respectively.
    pub fn fill(context: &GpuContext, shape: &Vec<usize>, value: f32) -> Self {
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
    pub fn zeros(context: &GpuContext, shape: &Vec<usize>) -> Self {
        assert_eq!(shape.len(), 2, "Shape must be length of 2 representing rows and columns respectively.");

        let size = shape[0] * shape[1];
        let data_gpu = context.get_stream().alloc_zeros::<f32>(size).expect("Cannot allocate memory in VRAM for this tensor.");
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

    pub fn get_data(&self) -> &CudaSlice<f32> {
        &self.data
    }

    /// Fetches `data` from the VRAM.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    pub fn download(&self, context: &GpuContext) -> Matrix {
        let mut mat = Matrix::new(self.rows(), self.cols());
        mat.v = download_cuda_slice(context.get_stream(), &self.data);
        mat
    }

    /// Sets all the elements of this tensor to a value in the VRAM.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `value` - The value to set.
    pub fn broadcast(&mut self, context: &GpuContext, value: f32) {
        context.gpu_memset(&self.data, value);
    }
}