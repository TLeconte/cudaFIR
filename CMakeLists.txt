cmake_minimum_required(VERSION 3.8)

# set the project name
project(CUDACONVOLVE C CUDA)

add_library(asound_module_pcm_cudaFIR SHARED src/cudaFIR_plugin.c src/convolution.cu)

set_target_properties(asound_module_pcm_cudaFIR PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(asound_module_pcm_cudaFIR PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(asound_module_pcm_cudaFIR asound cufft) 
set_target_properties(asound_module_pcm_cudaFIR PROPERTIES LINKER_LANGUAGE "CUDA")
set_property(TARGET asound_module_pcm_cudaFIR PROPERTY CUDA_ARCHITECTURES 53)

add_executable(cudaFIRtest src/cudaFIR_test.c src/convolution.cu)
set_target_properties(cudaFIRtest PROPERTIES LINKER_LANGUAGE "CUDA")
set_property(TARGET cudaFIRtest PROPERTY CUDA_ARCHITECTURES 53)
target_link_libraries(cudaFIRtest  cufft )


