use std::fmt::Debug;
use bytemuck::{NoUninit, Pod};
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use half::f16;
use crate::io::device::GpuContext;
use crate::util::core::Tensor;
use crate::util::log::Error;

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

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
}

// 3. Implement the trait explicitly for your two allowed types
impl PrecisionType for f32 {
    fn zero() -> Self { 0.0 }
    fn precision() -> Precision { Precision::FP32 }
    fn from_f32(v: f32) -> Self { v }
    fn to_f32(&self) -> f32 { *self }
    fn from_f16(v: f16) -> Self { v.to_f32()}
    fn to_f16(&self) -> f16 { f16::from_f32(*self) }
}

impl PrecisionType for f16 {
    fn zero() -> Self { f16::ZERO }
    fn precision() -> Precision { Precision::FP16 }
    fn from_f32(v: f32) -> Self { f16::from_f32(v) }
    fn to_f32(&self) -> f32 { f16::to_f32(*self) }
    fn from_f16(v: f16) -> Self { v }
    fn to_f16(&self) -> f16 { *self }
}

// Casting precisions for tensors
pub(crate) trait CastPrecision<Target> {
    fn cast(self, context: &GpuContext) -> Result<Target, Error>;
}

impl<T: PrecisionType, U: PrecisionType> CastPrecision<Tensor<U>> for Tensor<T> {
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
    fn cast(self, context: &GpuContext) -> Result<Tensor<U>, Error> {
        let tensor = Tensor::<U>::zeros(context, &[self.rows(), self.cols()]);
        context.gpu_cast_t(self.get_data(), tensor.get_data())?;
        Ok(tensor)
    }
}