#pragma once

#ifndef MATH_CUH
#define MATH_CUH

#include "types.cuh"

template <typename T> struct CudaMath;

template<>
struct CudaMath<f32_t> {
    __device__ static inline f32_t pow(const f32_t x, const f32_t y) { return powf(x, y); }
    __device__ static inline f32_t exp(const f32_t x) { return expf(x); }
    __device__ static inline f32_t log(const f32_t x) { return logf(x); }
    __device__ static inline f32_t sqrt(const f32_t x) { return sqrtf(x); }
    __device__ static inline f32_t rsqrt(const f32_t x) { return rsqrtf(x); }
    __device__ static inline f32_t tanh(const f32_t x) { return tanhf(x); }
    __device__ static inline f32_t abs(const f32_t x) { return fabsf(x); }
    __device__ static inline f32_t max(const f32_t a, const f32_t b) { return fmaxf(a, b); }
    __device__ static inline f32_t min(const f32_t a, const f32_t b) { return fminf(a, b); }
    __device__ static inline f32_t clamp(const f32_t a, const f32_t b, const f32_t c) { return fminf(fmaxf(a, b), c); }
};

template<>
struct CudaMath<f16_t> {
    // Upsampling to fp32 to prevent overflow

    __device__ static inline f16_t pow(const f16_t x, const f16_t y) {
        return __float2half(powf(__half2float(x), __half2float(y)));
    }

    __device__ static inline f16_t exp(const f16_t x) { return __float2half(expf(__half2float(x))); }
    __device__ static inline f16_t log(const f16_t x) { return __float2half(logf(__half2float(x))); }
    __device__ static inline f16_t sqrt(const f16_t x) { return __float2half(sqrtf(__half2float(x))); }
    __device__ static inline f16_t rsqrt(const f16_t x) { return __float2half(rsqrtf(__half2float(x))); }
    __device__ static inline f16_t tanh(const f16_t x) { return __float2half(tanhf(__half2float(x))); }
    __device__ static inline f16_t abs(const f16_t x) { return __habs(x); }
    __device__ static inline f16_t max(const f16_t a, const f16_t b) { return __hmax(a, b); }
    __device__ static inline f16_t min(const f16_t a, const f16_t b) { return __hmin(a, b); }
    __device__ static inline f16_t clamp(const f16_t a, const f16_t b, const f16_t c) {
        return __hmin(__hmax(a, b), c);
    }
};

__device__ inline f32_t dev_gen_random_f32_t(
    const uint32_t col, const uint32_t row, const uint32_t seed
) {
    uint32_t state = row * 1103515245 ^ col * 6364136223846793005U ^ seed;
    state = state ^ state >> 17;
    state = state * 1103515245 + 12345;
    state = state ^ state >> 15;
    return static_cast<f32_t>(state) / static_cast<f32_t>(0xFFFFFFFFU);
}

/**
 * The signum function.
 * @tparam T Either f32_t or f16_t.
 * @param x The input value.
 * @return Sign of x. If x < 0.0, returns -1.0. If x > 0.0, returns 1.0. If x == 0.0, returns 0.0.
 */
template<typename T> __device__ inline T dev_signum(const T x) {
    const T zero = static_cast<T>(0.0f);

    if (x > zero) return static_cast<T>(1.0f);
    if (x < zero) return static_cast<T>(-1.0f);
    return zero;
}

template<typename T> __device__ inline void broadcast_kernel(T* dst, const T v, const uint32_t len) {
    if (const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < len) {
        dst[idx] = v;
    }
}

/**
 * Single thread matrix multiplication.
 *
 * @tparam T Either f32_t or f16_t.
 *
* @return The value of the dot product of the vectors from the row of matrix A and the column of matrix B
 */
template<typename T> __device__ T dev_gemm(
    const T* __restrict__ a, const T* __restrict__ b,
    uint32_t m, uint32_t n, uint32_t p,
    uint32_t tile_dim, uint32_t row, uint32_t col
);

/**
 * Calculates C = A * B.
 * @tparam T Either f32_t or f16_t.
 * @param a Matrix A.
 * @param b Matrix B.
 * @param c Matrix C.
 * @param m Rows of matrix A.
 * @param n Columns of matrix A or rows of matrix B.
 * @param p Columns of matrix B.
 * @param tile_dim The dimension (both width and height) of the 2D thread block and data tile.
 */
template<typename T> __device__ void gemm_kernel(
    const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
    uint32_t m, uint32_t n, uint32_t p, uint32_t tile_dim
);

/**
 * Calculates C = A + B
 * @tparam T Either f32_t or f16_t.
 * @param a Matrix A.
 * @param b Matrix B.
 * @param c Matrix C.
 * @param m Rows of each matrix.
 * @param n Columns of each matrix.
 */
template<typename T>
__device__ void geam_kernel(const T * __restrict__ a, const T * __restrict__ b, T * __restrict__ c,
                                   uint32_t m, uint32_t n);

#endif
