#pragma once

#ifndef TYPES_HPP
#define TYPES_HPP

#include <cuda_runtime.h>
#include <cuda_fp16.h>

typedef __half f16_t;
typedef __half2 f16x2_t; // TODO implement later after f16_t
typedef float f32_t;
typedef double f64_t;

#endif