#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
// #include <cuSZp/cuSZp_entry_f32.h>
// #include <cuSZp/cuSZp_timer.h>
// #include <cuSZp/cuSZp_utility.h>
#include <cuSZp_entry_f32.h>
#include <cuSZp_timer.h>
#include <cuSZp_utility.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor compress(torch::Tensor input, float error_bound,
                       std::string mode) {
  CHECK_INPUT(input);
  // Get the input tensor's data pointer and size
  float *d_input_data = input.data_ptr<float>();
  int64_t num_elements = input.numel();
  size_t compressed_size = 0;

  // Cuda allocate memory for the compressed output
  unsigned char *d_compressed_data;
  cudaMalloc((void **)&d_compressed_data, num_elements * sizeof(float));
  cudaMemset(d_compressed_data, 0, num_elements * sizeof(float));

  // Initializing CUDA Stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Just a warmup.
  SZp_compress_deviceptr_f32(d_input_data, d_compressed_data, num_elements,
                             &compressed_size, error_bound, stream);
  // Ensure on a 4096 boundary
  // compressed_size = (compressed_size + 4095) / 4096 * 4096;
  // Create a new tensor on the GPU from the compressed output
  cudaStreamSynchronize(stream);
  torch::Tensor output = torch::empty(
      {compressed_size}, torch::TensorOptions()
                             .dtype(torch::kUInt8)
                             .device(torch::kCUDA)
                             .memory_format(torch::MemoryFormat::Contiguous));
  // write from d_compressed_data
  cudaMemcpy(output.data_ptr<unsigned char>(), d_compressed_data,
             compressed_size, cudaMemcpyDeviceToDevice);
  // Sync free
  cudaStreamSynchronize(stream);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // cudaMemGetInfo(&free_byte, &total_byte);
  // printf("GPU memory usage before output: used = %f, free = %f MB, total = %f
  // MB\n",
  //       (double)(total_byte - free_byte) / 1024.0 / 1024.0, (double)free_byte
  //       / 1024.0 / 1024.0, (double)total_byte / 1024.0 / 1024.0);
  cudaFree(d_compressed_data);
  cudaStreamDestroy(stream);
  CHECK_INPUT(output);
  return output;
}

torch::Tensor decompress(torch::Tensor compressed_data, int64_t num_elements,
                         size_t compressed_size, float error_bound,
                         std::string mode) {
  CHECK_INPUT(compressed_data);
  // Get the input tensor's data pointer and size
  unsigned char *d_compressed_data = compressed_data.data_ptr<unsigned char>();

  // torch::Tensor decompressed_data = torch::empty(
  //     , torch::TensorOptions()
  //                         .dtype(torch::kFloat32)
  //                         .device(torch::kCUDA)
  //                         .memory_format(torch::MemoryFormat::Contiguous));
  torch::Tensor decompressed_data = torch::zeros(
      {num_elements}, torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(torch::kCUDA)
                          .memory_format(torch::MemoryFormat::Contiguous));
  float *d_decompressed_data = decompressed_data.data_ptr<float>();

  // Initializing CUDA Stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  SZp_decompress_deviceptr_f32(d_decompressed_data, d_compressed_data,
                               num_elements, compressed_size, error_bound,
                               stream);
  cudaStreamSynchronize(stream);
  // Check cuda errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaStreamDestroy(stream);
  CHECK_INPUT(decompressed_data);
  return decompressed_data;
}

#define CUDA_CHECK_RETURN(value) {											\
    cudaError_t _m_cudaStat = value;										\
    if (_m_cudaStat != cudaSuccess) {										\
        fprintf(stderr, "Error %s at line %d in file %s\n",				\
                cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
        exit(1);															\
    } }

torch::Tensor compress_async(torch::Tensor input, torch::Tensor output, float error_bound,
                       std::string mode) {
  CHECK_INPUT(input);
  CHECK_INPUT(output);
  // Get the input tensor's data pointer and size
  at::cuda::CUDAStream torch_stream = at::cuda::getCurrentCUDAStream();
  float *d_input_data = input.data_ptr<float>();
  cudaStream_t stream = torch_stream.stream();
  int64_t num_elements = input.numel();
  torch::Tensor compressed_size = torch::zeros({1}, torch::TensorOptions()
                                                     .dtype(torch::kInt64)
                                                     .device(torch::kCPU)
                                                     .pinned_memory(true));
 int64_t *comp_size = compressed_size.data_ptr<int64_t>();

  // Cuda allocate memory for the compressed output
  unsigned char *d_compressed_data = output.data_ptr<unsigned char>();

  // Just a warmup.
  SZp_compress_deviceptr_f32_async(d_input_data, d_compressed_data, num_elements,
                             comp_size, error_bound, stream);
  // Ensure on a 4096 boundary
  // compressed_size = (compressed_size + 4095) / 4096 * 4096;
  // Create a new tensor on the GPU from the compressed output
  // CUDA_CHECK_RETURN(cudaStreamSynchronize(stream));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Compress CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Tensor CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  return compressed_size;
}

torch::Tensor decompress_async(torch::Tensor compressed_data, int64_t num_elements,
                         size_t compressed_size, float error_bound,
                         std::string mode) {
  CHECK_INPUT(compressed_data);
  // Get the input tensor's data pointer and size
  unsigned char *d_compressed_data = compressed_data.data_ptr<unsigned char>();

  // torch::Tensor decompressed_data = torch::empty(
  //     , torch::TensorOptions()
  //                         .dtype(torch::kFloat32)
  //                         .device(torch::kCUDA)
  //                         .memory_format(torch::MemoryFormat::Contiguous));
  torch::Tensor decompressed_data = torch::zeros(
      {num_elements}, torch::TensorOptions()
                          .dtype(torch::kFloat32)
                          .device(torch::kCUDA)
                          .memory_format(torch::MemoryFormat::Contiguous));
  float *d_decompressed_data = decompressed_data.data_ptr<float>();

  // Initializing CUDA Stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  SZp_decompress_deviceptr_f32(d_decompressed_data, d_compressed_data,
                               num_elements, compressed_size, error_bound,
                               stream);
  cudaStreamSynchronize(stream);
  // Check cuda errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  cudaStreamDestroy(stream);
  CHECK_INPUT(decompressed_data);
  return decompressed_data;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compress", &compress, "Compress a PyTorch tensor using cuSZp");
  m.def("compress_async", &compress_async, "Compress a PyTorch tensor using cuSZp");
  m.def("decompress", &decompress, "Decompress a PyTorch tensor using cuSZp");
  m.def("decompress_async", &decompress_async, "Decompress a PyTorch tensor using cuSZp");
}
