#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
  return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T from_float(float v);

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float v) {
  return __float2half(v);
}

template <typename T>
struct alignas(16) Pack8 {  // 16 bytes = 8 bf16/fp16
  T x[8];
};

template <typename T>
__device__ __forceinline__ Pack8<T> vec_load(const T* p) {
  return *reinterpret_cast<const Pack8<T>*>(p);
}

template <typename T>
__device__ __forceinline__ void vec_store(T* p, const Pack8<T>& v) {
  *reinterpret_cast<Pack8<T>*>(p) = v;
}

// ============================================================================
// MoE per-token expert combine: out[m, d] = sum_k input[m, k, d].
//
// Specialized on TOPK at compile time so the inner reduction unrolls cleanly.
// Vectorized uint4 (= 8 bf16) load/store for HBM-bandwidth efficiency.
// fp32 accumulator to keep numeric error bounded for bf16/fp16 inputs.
//
// Grid: (m, ceil(h / (kThreads * VEC))) — each block handles one (token,
// h-chunk). Inside the block, threads stride VEC = 8 bf16 each.
// ============================================================================
template <typename DType, int TOPK, int kThreads = 256>
__global__ void moe_sum_kernel(
    DType* __restrict__ out,             // (m, h)
    const DType* __restrict__ input,     // (m, TOPK, h)
    int h) {
  constexpr int VEC = 8;  // bf16/fp16: 8 elements per uint4 load
  static_assert(sizeof(DType) == 2, "kernel only for bf16/fp16");

  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int vec_base = blockIdx.x * kThreads * VEC + tid * VEC;
  if (vec_base >= h) return;

  // Per-row base: input[row * TOPK * h], out[row * h]
  const DType* in_row = input + static_cast<int64_t>(row) * TOPK * h;
  DType* out_row = out + static_cast<int64_t>(row) * h;

  float acc[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    const Pack8<DType> v = vec_load<DType>(in_row + k * h + vec_base);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc[i] += to_float<DType>(v.x[i]);
    }
  }

  Pack8<DType> out_v;
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    out_v.x[i] = from_float<DType>(acc[i]);
  }
  vec_store<DType>(out_row + vec_base, out_v);
}

// Generic-TOPK fallback for non-specialized values: same memory layout but
// runtime topk loop. Sacrifices a bit of code-gen quality but still beats
// torch.sum thanks to the vectorized loads.
template <typename DType, int kThreads = 256>
__global__ void moe_sum_kernel_generic(
    DType* __restrict__ out,
    const DType* __restrict__ input,
    int topk,
    int h) {
  constexpr int VEC = 8;
  static_assert(sizeof(DType) == 2);

  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int vec_base = blockIdx.x * kThreads * VEC + tid * VEC;
  if (vec_base >= h) return;

  const DType* in_row = input + static_cast<int64_t>(row) * topk * h;
  DType* out_row = out + static_cast<int64_t>(row) * h;

  float acc[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) acc[i] = 0.0f;

  for (int k = 0; k < topk; ++k) {
    const Pack8<DType> v = vec_load<DType>(in_row + k * h + vec_base);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc[i] += to_float<DType>(v.x[i]);
    }
  }

  Pack8<DType> out_v;
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    out_v.x[i] = from_float<DType>(acc[i]);
  }
  vec_store<DType>(out_row + vec_base, out_v);
}

template <typename T>
struct TypeTag {
  using type = T;
};

struct MoeSumKernel {
  static void run(
      const tvm::ffi::TensorView out,    // (m, h) bf16/fp16
      const tvm::ffi::TensorView input)  // (m, topk, h)
  {
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto m_ = SymbolicSize{"M"};
    auto k_ = SymbolicSize{"K"};
    auto h_ = SymbolicSize{"H"};

    TensorMatcher({m_, h_})
        .with_dtype<__nv_bfloat16, __half>(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(out);
    TensorMatcher({m_, k_, h_})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(input);

    const auto m = m_.unwrap();
    const auto topk = k_.unwrap();
    const auto h = h_.unwrap();
    if (m == 0 || h == 0 || topk == 0) {
      return;
    }
    RuntimeCheck(h % 8 == 0,
                 "moe_sum kernel requires h divisible by 8 (uint4 vec), got h=", h);
    constexpr int kThreads = 256;
    constexpr int VEC = 8;
    const auto blocks_x = static_cast<unsigned>((h + kThreads * VEC - 1) /
                                                (kThreads * VEC));
    const auto blocks_y = static_cast<unsigned>(m);
    const auto device = device_.unwrap();

    auto launch_specialized = [&](auto type_tag, auto topk_tag) {
      using DType = typename decltype(type_tag)::type;
      constexpr int TOPK = decltype(topk_tag)::value;
      LaunchKernel(dim3(blocks_x, blocks_y), kThreads, device)(
          moe_sum_kernel<DType, TOPK, kThreads>,
          static_cast<DType*>(out.data_ptr()),
          static_cast<const DType*>(input.data_ptr()),
          static_cast<int>(h));
    };
    auto launch_generic = [&](auto type_tag) {
      using DType = typename decltype(type_tag)::type;
      LaunchKernel(dim3(blocks_x, blocks_y), kThreads, device)(
          moe_sum_kernel_generic<DType, kThreads>,
          static_cast<DType*>(out.data_ptr()),
          static_cast<const DType*>(input.data_ptr()),
          static_cast<int>(topk),
          static_cast<int>(h));
    };

    auto dispatch = [&](auto type_tag) {
      if (topk == 2) {
        launch_specialized(type_tag, std::integral_constant<int, 2>{});
      } else if (topk == 4) {
        launch_specialized(type_tag, std::integral_constant<int, 4>{});
      } else if (topk == 8) {
        launch_specialized(type_tag, std::integral_constant<int, 8>{});
      } else if (topk == 6) {
        launch_specialized(type_tag, std::integral_constant<int, 6>{});
      } else if (topk == 16) {
        launch_specialized(type_tag, std::integral_constant<int, 16>{});
      } else {
        launch_generic(type_tag);
      }
    };

    if (data_dtype_.unwrap().code == DLDataTypeCode::kDLBfloat) {
      dispatch(TypeTag<__nv_bfloat16>{});
    } else {
      dispatch(TypeTag<__half>{});
    }
  }
};

}  // namespace
