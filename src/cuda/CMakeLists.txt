cmake_minimum_required(VERSION 3.18)
project(cuda_nn_math LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
    add_compile_definitions(__CUDA_ARCH__=800)
endif()

add_library(nn_math OBJECT nn_math.cu
        kernel/types.cuh
        kernel/util.cuh kernel/util.cu
        kernel/math.cuh kernel/math.cu
        kernel/ffn/forward.cuh kernel/ffn/forward.cu kernel/ffn/forward_wmma.cu
        kernel/ffn/backward.cuh kernel/ffn/backward.cu kernel/ffn/backward_wmma.cu
        kernel/conv/forward.cuh kernel/conv/forward.cu kernel/conv/forward_fft.cu)

set_target_properties(
        nn_math
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)