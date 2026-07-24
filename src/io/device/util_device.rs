use crate::io::device::GpuContext;
use crate::log::Error;
use crate::{Precision, PrecisionType};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

impl GpuContext {
    pub(crate) fn gpu_cast_t<T: PrecisionType, U: PrecisionType>(
        &self,
        src: &CudaSlice<T>,
        dst: &CudaSlice<U>,
    ) -> Result<(), Error> {
        if src.len() != dst.len() {
            return Err(Error::MismatchedDimensions {
                reason: "GPU precision cast source vs destination buffer sizes",
                expected: dst.len(),
                found: src.len(),
            });
        }

        let len = u32::try_from(src.len())?;

        let mut builder;
        if T::precision() == Precision::FP32 && U::precision() == Precision::FP16 {
            builder = self.stream.launch_builder(&self.cast_f32_f16_func);
        } else if T::precision() == Precision::FP16 && U::precision() == Precision::FP32 {
            builder = self.stream.launch_builder(&self.cast_f16_f32_func);
        } else {
            return Err(Error::UnsupportedTypeCast {
                from: T::precision(),
                to: U::precision(),
            });
        }

        builder.arg(src).arg(dst).arg(&len);

        let cfg = LaunchConfig {
            grid_dim: ((len + self.tile_dim_2_minus_1) / self.tile_dim_2, 1, 1),
            block_dim: (self.tile_dim_2, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            builder.launch(cfg)?;
        }

        Ok(())
    }

    pub(crate) fn gpu_broadcast<T: PrecisionType>(
        &self,
        dst: &CudaSlice<T>,
        v: T,
    ) -> Result<(), Error> {
        let len = dst.len();
        let len_u32 = u32::try_from(len)?;

        let mut builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.broadcast_func.0,
            Precision::FP16 => &self.broadcast_func.1,
        });
        builder.arg(dst).arg(&v).arg(&len_u32);

        let cfg = LaunchConfig {
            grid_dim: (
                (u32::try_from(len)? + self.tile_dim_2_minus_1) / self.tile_dim_2,
                1,
                1,
            ),
            block_dim: (self.tile_dim_2, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            builder.launch(cfg)?;
        }

        Ok(())
    }

    // Single-precision General Matrix Multiply
    pub(crate) fn gpu_matrix_mul<T: PrecisionType>(
        &self,
        a_dev: &CudaSlice<T>,
        m: usize,
        n: usize,
        b_dev: &CudaSlice<T>,
        p: usize,
        c_dev: &mut CudaSlice<T>,
    ) -> Result<(), Error> {
        let mut builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.gemm_func.0,
            Precision::FP16 => &self.gemm_func.1,
        });
        builder.arg(a_dev).arg(b_dev).arg(c_dev);
        builder.arg(&m).arg(&n).arg(&p);
        builder.arg(&self.tile_dim);

        let cfg =
            self.calculate_cfg2d(p, m, 2 * self.tile_dim_2 * u32::try_from(size_of::<T>())?)?;

        unsafe {
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        Ok(())
    }

    pub(crate) fn gpu_matrix_add<T: PrecisionType>(
        &self,
        a_dev: &CudaSlice<T>,
        m: usize,
        n: usize,
        b_dev: &CudaSlice<T>,
        c_dev: &mut CudaSlice<T>,
    ) -> Result<(), Error> {
        let mut builder = self.stream.launch_builder(match T::precision() {
            Precision::FP32 => &self.geam_func.0,
            Precision::FP16 => &self.geam_func.1,
        });
        builder.arg(a_dev).arg(b_dev).arg(c_dev);
        builder.arg(&m).arg(&n);

        let cfg = self.calculate_cfg2d(n, m, 0)?;

        unsafe {
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;

        Ok(())
    }
}
