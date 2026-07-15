use crate::core::{ImageBatch, Matrix, Tensor};
use crate::io::device::GpuContext;
use crate::util::log::Error;
use bytemuck::{NoUninit, Pod};
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use half::f16;
use std::fmt::Debug;

#[derive(Debug, PartialEq)]
pub enum Precision {
    /// 32-bit floating point
    FP32,
    /// 16-bit floating point
    FP16
}

impl Precision {
    pub fn size_bytes(&self) -> usize {
        match self {
            Precision::FP32 => size_of::<f32>(),
            Precision::FP16 => size_of::<f16>()
        }
    }
}

/// Wraps around [`f32`] and [`f16`]. These are the two types supported in the network.
pub trait PrecisionType: private::Sealed + DeviceRepr + Copy + 'static + ValidAsZeroBits + Pod + NoUninit + Debug {
    fn zero() -> Self;
    fn precision() -> Precision;
    fn from_f32(v: f32) -> Self;
    fn to_f32(&self) -> f32;
    fn from_f16(v: f16) -> Self;
    fn to_f16(&self) -> f16;
}

pub trait TensorContainerType: private::TensorSealed {
    type ShapeArray: Copy + Debug;
    type WithPrecision<U: PrecisionType>: TensorContainerType;
    fn shape_from_slice(slice: &[usize]) -> Self::ShapeArray;
}

mod private {
    use crate::core::{ImageBatch, Matrix};
    use crate::PrecisionType;

    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}

    pub trait TensorSealed {}
    impl<T: PrecisionType> TensorSealed for Vec<T> {}
    impl<T: PrecisionType> TensorSealed for Matrix<T> {}
    impl<T: PrecisionType> TensorSealed for ImageBatch<T> {}
}

// 3. Implement the trait explicitly for your two allowed types
impl PrecisionType for f32 {
    #[inline]
    fn zero() -> Self { 0.0 }
    #[inline]
    fn precision() -> Precision { Precision::FP32 }
    #[inline]
    fn from_f32(v: f32) -> Self { v }
    #[inline]
    fn to_f32(&self) -> f32 { *self }
    #[inline]
    fn from_f16(v: f16) -> Self { v.to_f32()}
    #[inline]
    fn to_f16(&self) -> f16 { f16::from_f32(*self) }
}

impl PrecisionType for f16 {
    #[inline]
    fn zero() -> Self { f16::ZERO }
    #[inline]
    fn precision() -> Precision { Precision::FP16 }
    #[inline]
    fn from_f32(v: f32) -> Self { f16::from_f32(v) }
    #[inline]
    fn to_f32(&self) -> f32 { f16::to_f32(*self) }
    #[inline]
    fn from_f16(v: f16) -> Self { v }
    #[inline]
    fn to_f16(&self) -> f16 { *self }
}

impl<T: PrecisionType> TensorContainerType for Vec<T> {
    type ShapeArray = [usize; 1];
    type WithPrecision<U: PrecisionType> = Vec<U>;
    fn shape_from_slice(slice: &[usize]) -> Self::ShapeArray { [slice[0]] }
}

impl<T: PrecisionType> TensorContainerType for Matrix<T> {
    type ShapeArray = [usize; 2];
    type WithPrecision<U: PrecisionType> = Matrix<U>;
    fn shape_from_slice(slice: &[usize]) -> Self::ShapeArray { [slice[0], slice[1]] }
}

impl<T: PrecisionType> TensorContainerType for ImageBatch<T> {
    type ShapeArray = [usize; 4];
    type WithPrecision<U: PrecisionType> = ImageBatch<U>;
    fn shape_from_slice(slice: &[usize]) -> Self::ShapeArray { [slice[0], slice[1], slice[2], slice[3]] }
}

// Casting precisions for tensors
pub(crate) trait CastPrecision<Target> {
    fn cast(self, context: &GpuContext) -> Result<Target, Error>;
}

impl<T: PrecisionType, U: PrecisionType, K: TensorContainerType> CastPrecision<Tensor<U, K::WithPrecision<U>>> for Tensor<T, K>
where
    K::ShapeArray: AsRef<[usize]>,
    <K::WithPrecision<U> as TensorContainerType>::ShapeArray: AsRef<[usize]>,
{
    /// Casts the tensor from type `T` to type `U`. New GPU buffer is allocated which is
    /// filled with the type-casted values. Then, the old GPU buffer is deallocated.
    ///
    /// # Returns
    /// The type-casted tensor.
    ///
    /// # Errors
    /// This function will return an error if:
    /// * [`Error::MismatchedDimensions`] - The allocation length of the `src` device buffer
    ///   does not match the allocated capacity of the destination `dst` buffer.
    /// * [`Error::UnsupportedTypeCast`] - The specified type parameter conversion matrix pairing
    ///   (from type $T$ to $U$) is unsupported by the underlying hardware kernel architecture.
    /// * [`Error::DriverError`] - An asynchronous hardware or synchronisation failure occurs while launching or executing
    ///   the type cast kernel on the GPU driver.
    fn cast(self, context: &GpuContext) -> Result<Tensor<U, K::WithPrecision<U>>, Error> {
        let tensor = Tensor::<U, K::WithPrecision<U>>::zeros(
            context,
            &<K::WithPrecision<U> as TensorContainerType>::shape_from_slice(&self.get_shape().as_ref())
        )?;
        
        context.gpu_cast_t(self.get_data(), tensor.get_data())?;
        self.free_and_sync(context)?;
        
        Ok(tensor)
    }
}

