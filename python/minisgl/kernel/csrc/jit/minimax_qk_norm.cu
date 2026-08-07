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

inline constexpr int kWarpSize = 32;
inline constexpr int kThreads = 256;
inline constexpr int kWarps = kThreads / kWarpSize;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_sum(float value, float* warp_sums) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;

  value = warp_sum(value);
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    value = lane < kWarps ? warp_sums[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads();
  return warp_sums[0];
}

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <>
__device__ __forceinline__ float to_float<__half>(__half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value);

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

template <>
__device__ __forceinline__ __half from_float<__half>(float value) {
  return __float2half(value);
}

template <typename DType>
__launch_bounds__(kThreads) __global__ void minimax_qk_cross_head_norm_kernel(
    DType* __restrict__ qkv,          // [T, q_dim + 2 * kv_dim]
    const DType* __restrict__ q_w,    // [q_dim]
    const DType* __restrict__ k_w,    // [kv_dim]
    int tokens,
    int qkv_stride,
    int q_dim,
    int kv_dim,
    float eps) {
  __shared__ float smem[kWarps];
  const int token = blockIdx.x;
  const bool is_k = blockIdx.y != 0;
  const int dim = is_k ? kv_dim : q_dim;
  const int offset = is_k ? q_dim : 0;
  DType* row = qkv + static_cast<size_t>(token) * qkv_stride + offset;
  const DType* weight = is_k ? k_w : q_w;

  float sum_sq = 0.0f;
  int i = threadIdx.x;
  for (; i + blockDim.x * 3 < dim; i += blockDim.x * 4) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float v = to_float<DType>(row[i + u * blockDim.x]);
      sum_sq += v * v;
    }
  }
  for (; i < dim; i += blockDim.x) {
    const float v = to_float<DType>(row[i]);
    sum_sq += v * v;
  }
  const float row_sum_sq = block_sum(sum_sq, smem);
  const float inv = rsqrtf(row_sum_sq / static_cast<float>(dim) + eps);

  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = to_float<DType>(row[i]);
    const float w = to_float<DType>(weight[i]);
    row[i] = from_float<DType>(v * inv * w);
  }
}

template <typename DType>
__launch_bounds__(kThreads) __global__ void minimax_qk_cross_head_norm_rope_store_kernel(
    DType* __restrict__ qkv,            // [T, q_dim + 2 * kv_dim]
    const DType* __restrict__ q_w,      // [q_dim]
    const DType* __restrict__ k_w,      // [kv_dim]
    const void* __restrict__ positions, // [T] int32/int64
    const float* __restrict__ cos_sin_cache,
    DType* __restrict__ k_cache,        // [capacity, kv_dim]
    DType* __restrict__ v_cache,        // [capacity, kv_dim]
    const void* __restrict__ out_loc,   // [T] int32/int64
    int tokens,
    int qkv_stride,
    int q_dim,
    int kv_dim,
    int head_dim,
    int rotary_dim,
    int cache_stride,
    long kv_cache_stride,
    bool positions_i64,
    bool loc_i64,
    float eps) {
  __shared__ float smem[kWarps];
  const int token = blockIdx.x;
  const bool is_k = blockIdx.y != 0;
  const int dim = is_k ? kv_dim : q_dim;
  const int offset = is_k ? q_dim : 0;
  DType* row = qkv + static_cast<size_t>(token) * qkv_stride + offset;
  const DType* weight = is_k ? k_w : q_w;

  float sum_sq = 0.0f;
  int i = threadIdx.x;
  for (; i + blockDim.x * 3 < dim; i += blockDim.x * 4) {
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      const float v = to_float<DType>(row[i + u * blockDim.x]);
      sum_sq += v * v;
    }
  }
  for (; i < dim; i += blockDim.x) {
    const float v = to_float<DType>(row[i]);
    sum_sq += v * v;
  }
  const float row_sum_sq = block_sum(sum_sq, smem);
  const float inv = rsqrtf(row_sum_sq / static_cast<float>(dim) + eps);
  const int heads = dim / head_dim;
  const int rotary_half = rotary_dim / 2;
  const long long pos = positions_i64
      ? static_cast<const int64_t*>(positions)[token]
      : static_cast<const int32_t*>(positions)[token];
  const float* rope = cos_sin_cache + pos * static_cast<long long>(cache_stride);
  const float* cos_ptr = rope;
  const float* sin_ptr = rope + rotary_half;
  DType* k_dst = nullptr;
  DType* v_dst = nullptr;
  DType* v_src = nullptr;
  if (is_k) {
    const long long cache_pos = loc_i64
        ? static_cast<const int64_t*>(out_loc)[token]
        : static_cast<const int32_t*>(out_loc)[token];
    k_dst = k_cache + cache_pos * kv_cache_stride;
    v_dst = v_cache + cache_pos * kv_cache_stride;
    v_src = qkv + static_cast<size_t>(token) * qkv_stride + q_dim + kv_dim;
  }

  // Process each rotary pair in one thread so the original low/high values are
  // read before either element is overwritten.
  const int rotary_pairs = heads * rotary_half;
  for (int p = threadIdx.x; p < rotary_pairs; p += blockDim.x) {
    const int head = p / rotary_half;
    const int j = p - head * rotary_half;
    const int lo = head * head_dim + j;
    const int hi = lo + rotary_half;
    const float x_lo = to_float<DType>(from_float<DType>(
        to_float<DType>(row[lo]) * inv * to_float<DType>(weight[lo])));
    const float x_hi = to_float<DType>(from_float<DType>(
        to_float<DType>(row[hi]) * inv * to_float<DType>(weight[hi])));
    const float c = cos_ptr[j];
    const float s = sin_ptr[j];
    const DType out_lo = from_float<DType>(x_lo * c - x_hi * s);
    const DType out_hi = from_float<DType>(x_hi * c + x_lo * s);
    row[lo] = out_lo;
    row[hi] = out_hi;
    if (is_k) {
      k_dst[lo] = out_lo;
      k_dst[hi] = out_hi;
    }
  }

  const int tail = head_dim - rotary_dim;
  if (tail > 0) {
    const int tail_elems = heads * tail;
    for (int p = threadIdx.x; p < tail_elems; p += blockDim.x) {
      const int head = p / tail;
      const int j = p - head * tail;
      const int idx = head * head_dim + rotary_dim + j;
      const DType out = from_float<DType>(
          to_float<DType>(row[idx]) * inv * to_float<DType>(weight[idx]));
      row[idx] = out;
      if (is_k) {
        k_dst[idx] = out;
      }
    }
  }

  if (!is_k) {
    return;
  }

  if ((kv_dim & 7) == 0) {
    const int vecs = kv_dim / 8;
    const auto v_src_vec = reinterpret_cast<const uint4*>(v_src);
    auto v_dst_vec = reinterpret_cast<uint4*>(v_dst);
    for (int v = threadIdx.x; v < vecs; v += blockDim.x) {
      v_dst_vec[v] = v_src_vec[v];
    }
  } else {
    for (int i = threadIdx.x; i < kv_dim; i += blockDim.x) {
      v_dst[i] = v_src[i];
    }
  }
}

template <typename T>
struct TypeTag {
  using type = T;
};

}  // namespace

struct MiniMaxQKNormKernel {
  static void run(
      const tvm::ffi::TensorView qkv,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      int q_dim,
      int kv_dim,
      double eps) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto tokens_ = SymbolicSize{"T"};
    auto qkv_dim_ = SymbolicSize{"D"};

    TensorMatcher({tokens_, qkv_dim_})
        .with_strides({-1, 1})
        .with_dtype<__nv_bfloat16, __half>(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(qkv);
    TensorMatcher({q_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(q_weight);
    TensorMatcher({kv_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(k_weight);
    RuntimeCheck(static_cast<int>(qkv_dim_.unwrap()) >= q_dim + 2 * kv_dim,
                 "qkv last dim is too small for MiniMax q/k norm");

    const auto tokens = static_cast<int>(tokens_.unwrap());
    if (tokens == 0) return;
    const auto device = device_.unwrap();
    const int qkv_stride = static_cast<int>(qkv.stride(0));

    auto launch = [&](auto type_tag) {
      using DType = typename decltype(type_tag)::type;
      LaunchKernel(dim3(tokens, 2, 1), kThreads, device)(
          minimax_qk_cross_head_norm_kernel<DType>,
          static_cast<DType*>(qkv.data_ptr()),
          static_cast<const DType*>(q_weight.data_ptr()),
          static_cast<const DType*>(k_weight.data_ptr()),
          tokens,
          qkv_stride,
          q_dim,
          kv_dim,
          static_cast<float>(eps));
    };

    if (data_dtype_.unwrap().code == DLDataTypeCode::kDLBfloat) {
      launch(TypeTag<__nv_bfloat16>{});
    } else {
      launch(TypeTag<__half>{});
    }
  }

  static void run_rope_store(
      const tvm::ffi::TensorView qkv,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView out_loc,
      int q_dim,
      int kv_dim,
      int head_dim,
      double eps) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto pos_dtype_ = SymbolicDType{};
    auto loc_dtype_ = SymbolicDType{};
    auto tokens_ = SymbolicSize{"T"};
    auto qkv_dim_ = SymbolicSize{"D"};
    auto rotary_dim_ = SymbolicSize{"R"};

    TensorMatcher({tokens_, qkv_dim_})
        .with_strides({-1, 1})
        .with_dtype<__nv_bfloat16, __half>(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(qkv);
    TensorMatcher({q_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(q_weight);
    TensorMatcher({kv_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(k_weight);
    TensorMatcher({tokens_})
        .with_dtype<int64_t, int32_t>(pos_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(positions);
    TensorMatcher({-1, rotary_dim_})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device_)
        .verify(cos_sin_cache);
    TensorMatcher({-1, kv_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(k_cache);
    TensorMatcher({-1, kv_dim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(v_cache);
    TensorMatcher({tokens_})
        .with_dtype<int64_t, int32_t>(loc_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(out_loc);

    const auto tokens = static_cast<int>(tokens_.unwrap());
    if (tokens == 0) return;
    const int qkv_dim = static_cast<int>(qkv_dim_.unwrap());
    const int rotary_dim = static_cast<int>(rotary_dim_.unwrap());
    RuntimeCheck(qkv_dim >= q_dim + 2 * kv_dim,
                 "qkv last dim is too small for MiniMax q/k norm+rope+store");
    RuntimeCheck(head_dim > 0 && q_dim % head_dim == 0 && kv_dim % head_dim == 0,
                 "MiniMax q/k norm+rope+store requires head-aligned q/kv dims");
    RuntimeCheck(rotary_dim > 0 && rotary_dim <= head_dim && (rotary_dim % 2) == 0,
                 "MiniMax q/k norm+rope+store requires an even rotary_dim <= head_dim");

    const auto device = device_.unwrap();
    const int qkv_stride = static_cast<int>(qkv.stride(0));
    const int cache_stride = static_cast<int>(cos_sin_cache.stride(0));
    const long kv_cache_stride = static_cast<long>(k_cache.stride(0));

    auto launch = [&](auto type_tag) {
      using DType = typename decltype(type_tag)::type;
      LaunchKernel(dim3(tokens, 2, 1), kThreads, device)(
          minimax_qk_cross_head_norm_rope_store_kernel<DType>,
          static_cast<DType*>(qkv.data_ptr()),
          static_cast<const DType*>(q_weight.data_ptr()),
          static_cast<const DType*>(k_weight.data_ptr()),
          positions.data_ptr(),
          static_cast<const float*>(cos_sin_cache.data_ptr()),
          static_cast<DType*>(k_cache.data_ptr()),
          static_cast<DType*>(v_cache.data_ptr()),
          out_loc.data_ptr(),
          tokens,
          qkv_stride,
          q_dim,
          kv_dim,
          head_dim,
          rotary_dim,
          cache_stride,
          kv_cache_stride,
          pos_dtype_.unwrap().bits == 64,
          loc_dtype_.unwrap().bits == 64,
          static_cast<float>(eps));
    };

    if (data_dtype_.unwrap().code == DLDataTypeCode::kDLBfloat) {
      launch(TypeTag<__nv_bfloat16>{});
    } else {
      launch(TypeTag<__half>{});
    }
  }
};
