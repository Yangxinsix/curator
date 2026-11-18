#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// The degree-three angular basis times amplitude and cutoff has degree five.
// One thread owns an output, including its entire sum. This also makes the
// transposed derivative contraction deterministic without scatter atomics.
template <typename T>
__global__ void evaluate_terms(const T* values, const int* offsets,
                               const int* indices, const T* coefficients,
                               int64_t size, int in_width, int out_width, T* result) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < size; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int output = i % out_width;
    const T* row = values + (i / out_width) * in_width;
    T sum = 0;
    for (int term = offsets[output]; term < offsets[output + 1]; ++term) {
      T product = coefficients[term];
      #pragma unroll
      for (int slot = 0; slot < 5; ++slot) {
        int column = indices[term * 5 + slot];
        if (column < 0) break;
        product *= row[column];
      }
      sum += product;
    }
    result[i] = sum;
  }
}

torch::Tensor evaluate(torch::Tensor x, torch::Tensor offsets, torch::Tensor indices,
                       torch::Tensor coefficients) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "expected contiguous CUDA matrix");
  TORCH_CHECK(offsets.device() == x.device() && indices.device() == x.device() &&
              coefficients.device() == x.device(), "plan and features must share a CUDA device");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt32 && indices.scalar_type() == torch::kInt32 &&
              coefficients.scalar_type() == x.scalar_type(), "invalid polynomial plan dtype");
  TORCH_CHECK(offsets.dim() == 1 && offsets.numel() > 1 && coefficients.dim() == 1 &&
              indices.dim() == 2 && indices.size(1) == 5 && indices.size(0) == coefficients.numel() &&
              offsets.is_contiguous() && indices.is_contiguous() && coefficients.is_contiguous(),
              "invalid polynomial plan dimensions");
  const c10::cuda::CUDAGuard guard(x.device());
  int out = offsets.numel() - 1;
  auto result = torch::empty({x.size(0), out}, x.options());
  if (result.numel()) {
    const int blocks = std::min<int64_t>(65535, (result.numel() + 127) / 128);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "dpa4c_small_polynomial", [&] {
      evaluate_terms<scalar_t><<<blocks, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<scalar_t>(), offsets.data_ptr<int>(), indices.data_ptr<int>(),
        coefficients.data_ptr<scalar_t>(), result.numel(), x.size(1), out,
        result.data_ptr<scalar_t>());
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("evaluate", &evaluate);
}
