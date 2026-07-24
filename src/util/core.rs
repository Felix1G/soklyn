use crate::io::device::GpuContext;
use crate::util::log::Error;
use crate::util::r#type::PrecisionType;
use crate::TensorContainerType;
use cudarc::driver::CudaSlice;
use std::sync::atomic::{AtomicUsize, Ordering};

#[macro_export]
macro_rules! getter {
    ($vis:vis $name:ident, $field:ident . $method:ident(), $t:ty) => {
        #[inline]
        $vis fn $name(&self) -> &$t {
            &self.$field.$method()
        }
    };

    ($vis:vis $name:ident, $($field:ident).+, $t:ty) => {
        #[inline]
        $vis fn $name(&self) -> &$t {
            &self.$($field).+
        }
    };
}

#[macro_export]
macro_rules! getter_option {
    ($vis:vis $name:ident, $($field:ident).+, Option<$inner:ty>) => {
        #[inline]
        $vis fn $name(&self) -> Option<&$inner> {
            self.$($field).+.as_ref()
        }
    };

    ($vis:vis $name:ident, $($field:ident).+, $ret:ty) => {
        #[inline]
        $vis fn $name(&self) -> &$ret {
            &self.$($field).+
        }
    };
}

#[macro_export]
macro_rules! getter_unwrap {
    ($vis:vis $name:ident, $field:ident . $method:ident(), $t:ty) => {
        #[inline]
        $vis fn $name(&self) -> &$t {
            &self.$field.$method().+.as_ref().unwrap()
        }
    };

    ($vis:vis $name:ident, $($field:ident).+, $t:ty) => {
        #[inline]
        $vis fn $name(&self) -> &$t {
            &self.$($field).+.as_ref().unwrap()
        }
    };
}

#[macro_export]
macro_rules! getter_copy {
    ($vis:vis $name:ident, $field:ident . $method:ident(), $t:ty) => {
        #[inline]
        $vis fn $name(&self) -> $t {
            self.$field.$method()
        }
    };

    ($vis:vis $name:ident, $($field:ident).+, $t:ty) => {
        #[inline]
        $vis fn $name(&self) -> $t {
            self.$($field).+
        }
    };
}

#[macro_export]
macro_rules! setter {
    ($vis:vis $name:ident, $($field:tt).+, $t:ty) => {
        #[inline]
        $vis fn $name(&mut self, val: $t) { self.$($field).+ = val; }
    };
}

static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn generate_unique_tensor_id() -> usize {
    TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn scramble_seed(step: u32, layer_id: u32) -> u32 {
    let mut hash = step.wrapping_mul(1_103_515_245).wrapping_add(layer_id);
    hash = (hash ^ (hash >> 16)).wrapping_mul(1_103_515_245);
    hash ^ (hash >> 15)
}

/// A typical matrix of size `(rows, cols)`.
#[derive(Debug)]
pub struct Matrix<T: PrecisionType> {
    rows: usize,
    cols: usize,
    pub v: Vec<T>, //values
}

impl<T: PrecisionType> Matrix<T> {
    /// Creates a new empty [`Matrix`] with dimensions set to `(0, 0)`.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(0, 0)
    }

    /// Creates a new zero-initialised [`Matrix`] with the given dimensions.
    ///
    /// # Arguments
    /// * `rows` - Number of rows.
    /// * `cols` - Number of columns.
    #[must_use]
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            v: vec![T::zero(); rows * cols],
        }
    }

    /// Creates a new [`Matrix`] where each element is initialised via a closure.
    ///
    /// The closure receives coordinates in the exact execution order: `(row, column)`.
    ///
    /// # Arguments
    /// * `init` - A closure mapping `(row, column)` to an initial value `T`.
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

    /// Safely fetches an immutable reference to the element at `(row, col)`.
    ///
    /// Returns `None` if any coordinates are out of bounds.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < self.rows && col < self.cols {
            self.v.get(row * self.cols + col)
        } else {
            None
        }
    }

    /// Safely fetches a mutable reference to the element at `(row, col)`.
    ///
    /// Returns `None` if any coordinates are out of bounds.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < self.rows && col < self.cols {
            self.v.get_mut(row * self.cols + col)
        } else {
            None
        }
    }

    /// Setting a new element at position `(row, col)`.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::IndexOutOfBounds`] - The coordinates are out of bounds.
    pub fn set(&mut self, row: usize, col: usize, val: T) -> Result<(), Error> {
        let idx = row * self.cols + col;
        let max_idx = self.rows * self.cols;

        if idx >= max_idx {
            return Err(Error::IndexOutOfBounds {
                idx,
                max_idx,
                reason: "setting element in matrix",
            });
        }

        self.v[idx] = val;

        Ok(())
    }

    /// Sets the value directly at the flat vector index `idx`.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::IndexOutOfBounds`] - The index is out of bounds.
    pub fn set_v(&mut self, idx: usize, val: T) -> Result<(), Error> {
        let max_idx = self.rows * self.cols;

        if idx >= max_idx {
            return Err(Error::IndexOutOfBounds {
                idx,
                max_idx,
                reason: "setting element in matrix",
            });
        }

        self.v[idx] = val;

        Ok(())
    }

    getter_copy!(pub rows, rows, usize);
    getter_copy!(pub cols, cols, usize);
    getter!(pub get_v, v, Vec<T>);

    /// Returns `true` if the container holds zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.v.len() == 0
    }

    /// Returns `true` if the container holds one or more elements.
    #[must_use]
    pub fn is_not_empty(&self) -> bool {
        !self.v.is_empty()
    }
}

/// A 4D data container representing a collection (batch) of multichannel 2D signals,
/// structured in a contiguous `NCHW` memory layout.
///
/// # Layout
/// * `N` is the number of batches.
/// * `C` is the number of channels.
/// * `H` is the height of the image.
/// * `W` is the width of the image.
#[derive(Debug)]
pub struct ImageBatch<T: PrecisionType> {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    v: Vec<T>, //values
}

impl<T: PrecisionType> ImageBatch<T> {
    /// Creates a new empty [`ImageBatch`] with dimensions set to `(0, 0, 0, 0)`.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            n: 0,
            c: 0,
            h: 0,
            w: 0,
            v: Vec::new(),
        }
    }

    /// Creates a new zero-initialised [`ImageBatch`] with the given dimensions.
    ///
    /// # Arguments
    /// * `n` - the number of batches.
    /// * `c` - the number of channels.
    /// * `h` - the height of the image.
    /// * `w` - the width of the image.
    #[must_use]
    pub fn new(n: usize, c: usize, h: usize, w: usize) -> Self {
        Self {
            n,
            c,
            h,
            w,
            v: vec![T::zero(); n * c * h * w],
        }
    }

    /// Creates a new [`ImageBatch`] where each element is initialised via a closure.
    ///
    /// The closure receives coordinates in the exact execution order: `(n, c, h, w)`.
    ///
    /// # Arguments
    /// * `init` - A closure mapping `(n, c, h, w)` to an initial value `T`.
    #[allow(clippy::many_single_char_names)]
    pub fn new_init<F>(n: usize, c: usize, h: usize, w: usize, mut init: F) -> Self
    where
        F: FnMut(usize, usize, usize, usize) -> T,
    {
        let total_size = n * c * h * w;
        let mut v = Vec::with_capacity(total_size);

        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        v.push(init(ni, ci, hi, wi));
                    }
                }
            }
        }

        Self { n, c, h, w, v }
    }

    /// Computes the flat 1D index corresponding to the given 4D coordinates.
    #[inline]
    #[must_use]
    pub fn idx(&self, n: usize, c: usize, h: usize, w: usize) -> usize {
        n * (self.c * self.h * self.w) + c * (self.h * self.w) + h * self.w + w
    }

    /// Safely fetches an immutable reference to the element at index `idx`.
    ///
    /// Returns `None` if any coordinates are out of bounds.
    #[must_use]
    pub fn get_v(&self, idx: usize) -> Option<&T> {
        if idx <= self.v.len() {
            self.v.get(idx)
        } else {
            None
        }
    }

    /// Safely fetches an immutable reference to the element at `(n, c, h, w)`.
    ///
    /// Returns `None` if any coordinates are out of bounds.
    #[must_use]
    pub fn get(&self, n: usize, c: usize, h: usize, w: usize) -> Option<&T> {
        if n < self.n && c < self.c && h < self.h && w < self.w {
            self.v.get(self.idx(n, c, h, w))
        } else {
            None
        }
    }

    /// Safely fetches a mutable reference to the element at `(n, c, h, w)`.
    ///
    /// Returns `None` if any coordinates are out of bounds.
    pub fn get_mut(&mut self, n: usize, c: usize, h: usize, w: usize) -> Option<&mut T> {
        if n < self.n && c < self.c && h < self.h && w < self.w {
            let idx = self.idx(n, c, h, w);
            self.v.get_mut(idx)
        } else {
            None
        }
    }

    /// Sets the value at the coordinate `(n, c, h, w)`.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::IndexOutOfBounds`] - The coordinates are out of bounds.
    pub fn set(&mut self, n: usize, c: usize, h: usize, w: usize, val: T) -> Result<(), Error> {
        if n >= self.n || c >= self.c || h >= self.h || w >= self.w {
            let idx = self.idx(n, c, h, w);
            let max_idx = self.n * self.c * self.h * self.w;
            return Err(Error::IndexOutOfBounds {
                idx,
                max_idx,
                reason: "reading element from image batch",
            });
        }

        let idx = self.idx(n, c, h, w);
        self.v[idx] = val;
        Ok(())
    }

    /// Sets the value directly at the flat vector index `idx`.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::IndexOutOfBounds`] - The index is out of bounds.
    pub fn set_v(&mut self, idx: usize, val: T) -> Result<(), Error> {
        let max_idx = self.n * self.c * self.h * self.w;

        if idx >= max_idx {
            return Err(Error::IndexOutOfBounds {
                idx,
                max_idx,
                reason: "setting element in image batch",
            });
        }

        self.v[idx] = val;

        Ok(())
    }

    getter_copy!(pub get_n, n, usize);
    getter_copy!(pub get_c, c, usize);
    getter_copy!(pub get_h, h, usize);
    getter_copy!(pub get_w, w, usize);
    getter!(pub get_data, v, Vec<T>);

    /// Returns `true` if the container holds zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.v.len() == 0
    }

    /// Returns `true` if the container holds one or more elements.
    #[must_use]
    pub fn is_not_empty(&self) -> bool {
        !self.v.is_empty()
    }
}

/// The multidimensional data container which generalises scalars, vectors, and matrices to
/// any number of dimensions—that serves as the fundamental memory and
/// mathematical tracking unit of a neural network.
///
/// Stores some variables in the live GPU memory buffers as `CudaSlice<f32>`. Data is generally row-major.
#[derive(Debug)]
pub struct Tensor<T: PrecisionType, K: TensorContainerType> {
    id: usize,
    shape: K::ShapeArray,
    data: CudaSlice<T>,
}

/// A 1-dimensional GPU tensor representing flat, contiguous data.
///
/// This is the device memory counterpart to a standard [`Vec`].
pub type Tensor1D<T> = Tensor<T, Vec<T>>;
/// A 2-dimensional GPU tensor structured in rows and columns.
///
/// This serves as the accelerated device counterpart to the host-side [`Matrix`].
pub type Tensor2D<T> = Tensor<T, Matrix<T>>;
/// A 4-dimensional GPU tensor organised in the `NCHW` layout.
///
/// This serves as the accelerated device counterpart to the host-side [`ImageBatch`]
/// and is optimised for convolutional network blocks.
pub type Tensor4D<T> = Tensor<T, ImageBatch<T>>;

impl<T: PrecisionType> Tensor1D<T> {
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape[0]
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fetches `data` from the live GPU memory buffers.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    ///
    /// # Errors
    /// Return an [`Error`] if data is unable to be downloaded from the GPU.
    pub fn download(&self, context: &GpuContext) -> Result<Vec<T>, Error> {
        Ok(context.get_stream().clone_dtoh(&self.data)?)
    }
}

impl<T: PrecisionType> Tensor2D<T> {
    #[inline]
    #[must_use]
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    #[must_use]
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    /// Fetches `data` from the live GPU memory buffers.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    ///
    /// # Errors
    /// Return an [`Error`] if data is unable to be downloaded from the GPU.
    pub fn download(&self, context: &GpuContext) -> Result<Matrix<T>, Error> {
        let mut mat = Matrix::new(self.rows(), self.cols());
        mat.v = context.get_stream().clone_dtoh(&self.data)?;
        Ok(mat)
    }
}

impl<T: PrecisionType> Tensor4D<T> {
    /// Returns the size of the outermost dimension (Axis 0).
    ///
    /// Corresponds to batch size, or the number of output filters for weights.
    #[inline]
    #[must_use]
    pub fn outer_dim(&self) -> usize {
        self.shape[0]
    }

    /// Semantic alias for `outer_dim()`.
    #[inline]
    #[must_use]
    pub fn batches(&self) -> usize {
        self.outer_dim()
    }

    /// Semantic alias for `outer_dim()`.
    #[inline]
    #[must_use]
    pub fn filters(&self) -> usize {
        self.outer_dim()
    }

    #[inline]
    #[must_use]
    pub fn channels(&self) -> usize {
        self.shape[1]
    }

    #[inline]
    #[must_use]
    pub fn height(&self) -> usize {
        self.shape[2]
    }

    #[inline]
    #[must_use]
    pub fn width(&self) -> usize {
        self.shape[3]
    }

    /// Fetches `data` from the live GPU memory buffers.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    ///
    /// # Errors
    /// Return an [`Error`] if data is unable to be downloaded from the GPU.
    pub fn download(&self, context: &GpuContext) -> Result<ImageBatch<T>, Error> {
        let mut img = ImageBatch::new(self.batches(), self.channels(), self.height(), self.width());
        img.v = context.get_stream().clone_dtoh(&self.data)?;
        Ok(img)
    }
}

impl<T: PrecisionType, K: TensorContainerType> Tensor<T, K>
where
    K::ShapeArray: AsRef<[usize]>,
{
    /// Creates a new tensor from CPU data vector.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `cpu_data` - The data value of the tensor.
    /// * `shape` - The shape of the matrix of this tensor.
    ///
    /// # Panics
    /// Panics if length of `cpu_data` does not match total elements of the tensor of size `shape`.
    ///
    /// # Errors
    /// Return an [`Error`] if data is unable to be uploaded to the GPU.
    pub fn from_cpu_vector(
        context: &GpuContext,
        cpu_data: &[T],
        shape: &K::ShapeArray,
    ) -> Result<Self, Error> {
        let size: usize = shape.as_ref().iter().product();
        if size != cpu_data.len() {
            return Err(Error::MismatchedDimensions {
                reason: "length of CPU data does not match the total elements of the tensor",
                expected: size,
                found: cpu_data.len(),
            });
        }

        let data_gpu = context.get_stream().clone_htod(cpu_data)?;
        context.get_stream().synchronize()?;

        Ok(Self {
            id: generate_unique_tensor_id(),
            shape: *shape,
            data: data_gpu,
        })
    }

    #[allow(unused)]
    pub(crate) fn from_gpu_slice(
        gpu_slice: CudaSlice<T>,
        shape: &K::ShapeArray,
    ) -> Result<Self, Error> {
        let size: usize = shape.as_ref().iter().product();
        if size != gpu_slice.len() {
            return Err(Error::MismatchedDimensions {
                reason: "length of CUDA slice does not match the total elements of the tensor",
                expected: gpu_slice.len(),
                found: size,
            });
        }

        Ok(Self {
            id: generate_unique_tensor_id(),
            shape: *shape,
            data: gpu_slice,
        })
    }

    /// Creates a new tensor with all the same values.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `shape` - The shape of the matrix of this tensor.
    /// * `value` - The value to be broadcasted to the entire tensor.
    ///
    /// # Errors
    /// Return an [`Error`] if data is unable to be broadcasted.
    pub fn fill(context: &GpuContext, shape: &K::ShapeArray, value: T) -> Result<Self, Error> {
        let mut tensor = Self::zeros(context, shape)?;
        tensor.broadcast(context, value)?;
        Ok(tensor)
    }

    /// Creates a new empty tensor.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `shape` - The shape of the matrix of this tensor.
    ///
    /// For 1D, it is the length of the array. Pass a `[usize; 1]`.
    ///
    /// For 2D, it is the rows and columns of the matrix respectively. Pass a `[usize; 2]`.
    ///
    /// For 4D, it is in `NCHW` format. Pass a `[usize; 4]`.
    ///
    /// # Panics
    /// Panics if the length of `shape` is not 2, which represents rows and column respectively.
    ///
    /// # Errors
    /// Return an [`Error`] if memory on the GPU is unable to be allocated.
    pub fn zeros(context: &GpuContext, shape: &K::ShapeArray) -> Result<Self, Error> {
        let size: usize = shape.as_ref().iter().product();
        let data_gpu = context.get_stream().alloc_zeros::<T>(size)?;
        context.get_stream().synchronize()?;

        Ok(Self {
            id: generate_unique_tensor_id(),
            shape: *shape,
            data: data_gpu,
        })
    }

    /// Each tensor possesses its own unique ID.
    pub fn get_id(&self) -> usize {
        self.id
    }

    getter!(pub get_data, data, CudaSlice<T>);
    getter!(pub get_shape, shape, K::ShapeArray);

    /// Safely frees this tensor from memory.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`].
    ///
    /// # Errors
    /// Return an [`Error`] if the CUDA stream is unable to synchronise.
    pub fn free_and_sync(self, context: &GpuContext) -> Result<(), Error> {
        drop(self);
        context.get_stream().synchronize()?;
        Ok(())
    }

    /// Sets all the elements of this tensor to a value in the live GPU memory buffers.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    /// * `value` - The value to set.
    ///
    /// # Errors
    /// Return an [`Error`] if data is unable to be broadcasted.
    pub fn broadcast(&mut self, context: &GpuContext, value: T) -> Result<(), Error> {
        context.gpu_broadcast(&self.data, value)
    }

    /// Copies this tensor by copying the data from the live GPU memory buffers.
    ///
    /// # Arguments
    /// * `context` - See [`GpuContext`]
    ///
    /// # Errors
    /// Return an [`Error`] if the data is unable to be copied in the GPU.
    pub fn clone(&self, context: &GpuContext) -> Result<Self, Error> {
        Ok(Self {
            id: generate_unique_tensor_id(),
            shape: self.shape,
            data: context.get_stream().clone_dtod(self.get_data())?,
        })
    }
}
