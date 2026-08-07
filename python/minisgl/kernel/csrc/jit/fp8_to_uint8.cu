#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Vectorized 16-byte copy of FP8 (uint8 in raw bytes) into a uint8 tensor.
// Pure data move — kept as a JIT kernel so callers don't need a separate
// torch extension build. Tail handled by a scalar epilogue.
__global__ void fp8_to_uint8_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int64_t n) {
  constexpr int kVecBytes = 16;
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  const int64_t vec_n = n / kVecBytes;
  for (int64_t i = tid; i < vec_n; i += stride) {
    *reinterpret_cast<uint4*>(dst + i * kVecBytes) =
        *reinterpret_cast<const uint4*>(src + i * kVecBytes);
  }
  // Scalar tail
  const int64_t tail_start = vec_n * kVecBytes;
  for (int64_t i = tail_start + tid; i < n; i += stride) {
    dst[i] = src[i];
  }
}

struct Fp8ToUint8Kernel {
  static void run(
      const tvm::ffi::TensorView src,  // any shape, uint8 bytes (caller views fp8 as uint8)
      const tvm::ffi::TensorView dst) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto n_ = SymbolicSize{"N"};
    TensorMatcher({n_})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(src);
    TensorMatcher({n_})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(dst);
    const auto n = n_.unwrap();
    if (n == 0) {
      return;
    }
    const auto device = device_.unwrap();
    constexpr int kThreads = 256;
    constexpr int kMaxBlocks = 1024;
    const int64_t vec_n = (n + 15) / 16;
    const auto blocks = static_cast<unsigned>(
        vec_n < kMaxBlocks * kThreads
            ? (vec_n + kThreads - 1) / kThreads
            : kMaxBlocks);
    LaunchKernel(blocks, static_cast<unsigned>(kThreads), device)(
        fp8_to_uint8_kernel,
        static_cast<const uint8_t*>(src.data_ptr()),
        static_cast<uint8_t*>(dst.data_ptr()),
        n);
  }
};

}  // namespace
