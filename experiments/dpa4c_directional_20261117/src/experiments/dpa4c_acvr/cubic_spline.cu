#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

// Cubic cardinal bases on the checkpoint's unit-spaced grid [-4,...,4].
// x = tanh(normalized feature). One thread evaluates all five basis values.
template <typename T>
__global__ void cubic(const T* x, T* out, int64_t n, int order) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const T value = x[i];
    const T cell = floor(value);
    const int first = static_cast<int>(cell) + 1;
    const T t = value - cell;
    const T u = 1 - t;
    T b[4];
    if (order == 0) {
      b[0] = u*u*u/6;
      b[1] = (3*t*t*t - 6*t*t + 4)/6;
      b[2] = (-3*t*t*t + 3*t*t + 3*t + 1)/6;
      b[3] = t*t*t/6;
    } else if (order == 1) {
      b[0] = -u*u/2;
      b[1] = (T(1.5)*t - 2)*t;
      b[2] = (-T(1.5)*t + 1)*t + T(0.5);
      b[3] = t*t/2;
    } else if (order == 2) {
      b[0] = u; b[1] = 3*t-2; b[2] = 1-3*t; b[3] = t;
    } else {
      b[0] = -1; b[1] = 3; b[2] = -3; b[3] = 1;
    }
    #pragma unroll
    for (int j = 0; j < 5; ++j) {
      const int slot = j-first;
      out[i*5+j] = (slot>=0 && slot<4) ? b[slot] : T(0);
    }
  }
}

torch::Tensor evaluate(torch::Tensor x, int64_t order) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "expected contiguous CUDA input");
  TORCH_CHECK(0 <= order && order <= 3, "derivative order must be 0..3");
  const c10::cuda::CUDAGuard guard(x.device());
  auto dims = x.sizes().vec();
  dims.push_back(5);
  auto out = at::empty(dims, x.options());
  if (x.numel() > 0) {
    const int blocks = std::min<int64_t>(65535, (x.numel()+255)/256);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "acvr_cubic_spline", [&] {
      cubic<scalar_t><<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), x.numel(), order);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("evaluate", &evaluate); }
