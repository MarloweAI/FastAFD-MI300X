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

struct QKNormRopeParams {
  const void* positions;
  void* q_ptr;
  void* k_ptr;
  const void* q_weight_ptr;
  const void* k_weight_ptr;
  const float* cos_sin_cache;
  int q_stride_t;
  int k_stride_t;
  int cache_stride_t;
  int q_rows;
  int k_rows;
  bool positions_i64;
  float eps;
  // Optional fused KV-cache store (replaces the separate store_kv_cache
  // launch): roped K rows and raw V rows land at kv_cache[out_loc[token]].
  void* k_cache;
  void* v_cache;
  const void* v_ptr;
  const void* out_loc;
  long kv_cache_stride_t;  // elements per cache token row
  int v_stride_t;
  bool loc_i64;
  bool store_kv;
};

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

// 8-byte vectorized pack of 4 bf16 / 4 fp16 elements.
template <typename T>
struct alignas(8) Pack4 {
  T x[4];
};

// Fused RMSNorm(Q, K) + NeoX RoPE in a single kernel. Ported from vLLM's
// csrc/fused_qknorm_rope_kernel.cu (TensorRT-LLM design) with these
// differences:
//   - takes separate Q and K split views instead of one combined QKV tensor;
//   - NeoX (interleave=false) only; cos_sin_cache laid out as
//     [max_position, head_dim] with first kHalf cos then kHalf sin.
//
// Each warp processes one (token, head) row. Each lane vec-loads 4 contiguous
// bf16 (uint2 = 8 bytes), warp-reduces sum-of-squares, applies norm in
// registers, then uses __shfl_xor with mask kPairOffset = kHalf/kElemsPerThread
// to swap rotation partners between lane k and lane k^kPairOffset, and writes
// back vectorized.
template <typename DType, int kQHeads, int kKVHeads, int kHeadDim, int kBlockM>
__global__ void //
qk_norm_rope_kernel(const __grid_constant__ QKNormRopeParams params) {
  static_assert(kHeadDim % 64 == 0, "kHeadDim must be a multiple of 64");
  constexpr int kHalf = kHeadDim / 2;
  constexpr int kElemsPerThread = kHeadDim / 32;
  constexpr int kPairOffset = kHalf / kElemsPerThread;
  static_assert(kElemsPerThread == 4, "this kernel assumes head_dim=128");

  const int warp_id = static_cast<int>(threadIdx.y);
  const int lane = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.x) * kBlockM + warp_id;
  const int total_rows =
      params.q_rows + params.k_rows + (params.store_kv ? params.k_rows : 0);
  if (row >= total_rows) {
    return;
  }
  if (row >= params.q_rows + params.k_rows) {
    // V row: plain vectorized copy into the cache (no norm / rope)
    const int rel = row - params.q_rows - params.k_rows;
    const int token = rel / kKVHeads;
    const int head = rel - token * kKVHeads;
    const long long pos = params.loc_i64
        ? static_cast<const int64_t*>(params.out_loc)[token]
        : static_cast<const int32_t*>(params.out_loc)[token];
    const DType* src = static_cast<const DType*>(params.v_ptr) +
                       token * params.v_stride_t + head * kHeadDim;
    DType* dst = static_cast<DType*>(params.v_cache) +
                 pos * params.kv_cache_stride_t + head * kHeadDim;
    reinterpret_cast<Pack4<DType>*>(dst)[lane] =
        reinterpret_cast<const Pack4<DType>*>(src)[lane];
    return;
  }
  const bool is_q = row < params.q_rows;
  const int rel_row = is_q ? row : (row - params.q_rows);
  const int heads = is_q ? kQHeads : kKVHeads;
  const int stride_t = is_q ? params.q_stride_t : params.k_stride_t;
  DType* base = static_cast<DType*>(is_q ? params.q_ptr : params.k_ptr);
  const DType* weight =
      static_cast<const DType*>(is_q ? params.q_weight_ptr : params.k_weight_ptr);

  const int token = rel_row / heads;
  const int head = rel_row - token * heads;
  DType* row_ptr = base + token * stride_t + head * kHeadDim;

  // Vectorized 8-byte load: 4 bf16 contiguous per lane.
  const Pack4<DType> v_in = reinterpret_cast<const Pack4<DType>*>(row_ptr)[lane];
  const Pack4<DType> v_w = reinterpret_cast<const Pack4<DType>*>(weight)[lane];

  float elems[kElemsPerThread];
  float sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    elems[i] = to_float<DType>(v_in.x[i]);
    sq += elems[i] * elems[i];
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sq += __shfl_xor_sync(0xffffffffu, sq, off);
  }
  const float inv = rsqrtf(sq / static_cast<float>(kHeadDim) + params.eps);

  // Apply RMSNorm finalize (× inv × weight).
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    elems[i] *= inv * to_float<DType>(v_w.x[i]);
  }

  // NeoX RoPE: pair partner is lane^kPairOffset (shfl_xor by 16 for kHeadDim=128).
  // For lane < kPairOffset (low half), pair is at higher position; negate
  // shuffled value so the formula becomes  y_lo = x_lo·cos − x_hi·sin.
  // For lane >= kPairOffset (high half), formula is y_hi = x_hi·cos + x_lo·sin.
  const int64_t pos = params.positions_i64
      ? static_cast<const int64_t*>(params.positions)[token]
      : static_cast<const int32_t*>(params.positions)[token];
  const float* cos_ptr = params.cos_sin_cache +
                         static_cast<long long>(pos) * params.cache_stride_t;
  const float* sin_ptr = cos_ptr + kHalf;

  float elems2[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    elems2[i] = __shfl_xor_sync(0xffffffffu, elems[i], kPairOffset);
    if (lane < kPairOffset) {
      elems2[i] = -elems2[i];
    }
  }
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    // dim_idx = lane*N + i maps to half-frequency index via (dim_idx*2) % kHeadDim / 2.
    // For low half (dim_idx < kHalf): half_dim = dim_idx.
    // For high half (dim_idx >= kHalf): half_dim = dim_idx - kHalf.
    const int dim_idx = lane * kElemsPerThread + i;
    const int half_dim = (dim_idx * 2) % kHeadDim / 2;
    const float c = cos_ptr[half_dim];
    const float s = sin_ptr[half_dim];
    elems[i] = elems[i] * c + elems2[i] * s;
  }

  // Vectorized 8-byte store back.
  Pack4<DType> v_out;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    v_out.x[i] = from_float<DType>(elems[i]);
  }
  reinterpret_cast<Pack4<DType>*>(row_ptr)[lane] = v_out;

  // Fused KV store: roped K also lands in the cache row for this token
  if (!is_q && params.store_kv) {
    const long long cpos = params.loc_i64
        ? static_cast<const int64_t*>(params.out_loc)[token]
        : static_cast<const int32_t*>(params.out_loc)[token];
    DType* dst = static_cast<DType*>(params.k_cache) +
                 cpos * params.kv_cache_stride_t + head * kHeadDim;
    reinterpret_cast<Pack4<DType>*>(dst)[lane] = v_out;
  }
}

template <typename T>
struct TypeTag {
  using type = T;
};

template <int kQHeads, int kKVHeads, int kHeadDim, int kBlockM = 16>
struct QKNormRopeKernel {
  // q, k          : strided (tokens, ?), the slice we get from qkv.split(...).
  // q_weight,
  // k_weight      : (head_dim,) in same dtype as q/k.
  // positions     : (tokens,) int64.
  // cos_sin_cache : (max_position, head_dim) fp32, layout [cos..., sin...].
  // eps           : RMSNorm epsilon.
  static void run(
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      double eps) {
    using namespace host;

    auto device_ = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto positions_dtype_ = SymbolicDType{};
    auto tokens_ = SymbolicSize{"T"};

    TensorMatcher({tokens_})
        .with_dtype<int64_t, int32_t>(positions_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(positions);
    TensorMatcher({tokens_, kQHeads * kHeadDim})
        .with_strides({-1, 1})
        .with_dtype<__nv_bfloat16, __half>(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(q);
    TensorMatcher({tokens_, kKVHeads * kHeadDim})
        .with_strides({-1, 1})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(k);
    TensorMatcher({kHeadDim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(q_weight);
    TensorMatcher({kHeadDim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(k_weight);
    TensorMatcher({-1, kHeadDim})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device_)
        .verify(cos_sin_cache);

    const auto tokens = static_cast<int>(tokens_.unwrap());
    if (tokens == 0) {
      return;
    }
    const auto device = device_.unwrap();
    const auto q_rows = tokens * kQHeads;
    const auto k_rows = tokens * kKVHeads;

    auto launch = [&](auto type_tag) {
      using DType = typename decltype(type_tag)::type;
      const QKNormRopeParams params{
          .positions = positions.data_ptr(),
          .q_ptr = q.data_ptr(),
          .k_ptr = k.data_ptr(),
          .q_weight_ptr = q_weight.data_ptr(),
          .k_weight_ptr = k_weight.data_ptr(),
          .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
          .q_stride_t = static_cast<int>(q.stride(0)),
          .k_stride_t = static_cast<int>(k.stride(0)),
          .cache_stride_t = static_cast<int>(cos_sin_cache.stride(0)),
          .q_rows = q_rows,
          .k_rows = k_rows,
          .positions_i64 = positions_dtype_.unwrap().bits == 64,
          .eps = static_cast<float>(eps),
      };
      const auto blocks =
          static_cast<unsigned>((q_rows + k_rows + kBlockM - 1) / kBlockM);
      LaunchKernel(blocks, dim3(32, kBlockM, 1), device)(
          qk_norm_rope_kernel<DType, kQHeads, kKVHeads, kHeadDim, kBlockM>,
          params);
    };

    if (data_dtype_.unwrap().code == DLDataTypeCode::kDLBfloat) {
      launch(TypeTag<__nv_bfloat16>{});
    } else {
      launch(TypeTag<__half>{});
    }
  }

  // Fused variant: + V passthrough and KV-cache store (replaces the separate
  // store_kv_cache launch).  k_cache/v_cache are the layer's cache viewed as
  // (capacity, kKVHeads*kHeadDim); out_loc is (tokens,) int32/int64.
  static void run_store(
      const tvm::ffi::TensorView positions,
      const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView q_weight,
      const tvm::ffi::TensorView k_weight,
      const tvm::ffi::TensorView cos_sin_cache,
      const tvm::ffi::TensorView k_cache,
      const tvm::ffi::TensorView v_cache,
      const tvm::ffi::TensorView out_loc,
      double eps) {
    using namespace host;

    auto device_ = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto positions_dtype_ = SymbolicDType{};
    auto loc_dtype_ = SymbolicDType{};
    auto tokens_ = SymbolicSize{"T"};

    TensorMatcher({tokens_})
        .with_dtype<int64_t, int32_t>(positions_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(positions);
    TensorMatcher({tokens_, kQHeads * kHeadDim})
        .with_strides({-1, 1})
        .with_dtype<__nv_bfloat16, __half>(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(q);
    TensorMatcher({tokens_, kKVHeads * kHeadDim})
        .with_strides({-1, 1})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(k);
    TensorMatcher({tokens_, kKVHeads * kHeadDim})
        .with_strides({-1, 1})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(v);
    TensorMatcher({kHeadDim}).with_dtype(data_dtype_).with_device<kDLCUDA>(device_).verify(q_weight);
    TensorMatcher({kHeadDim}).with_dtype(data_dtype_).with_device<kDLCUDA>(device_).verify(k_weight);
    TensorMatcher({-1, kHeadDim}).with_dtype<float>().with_device<kDLCUDA>(device_).verify(cos_sin_cache);
    TensorMatcher({-1, kKVHeads * kHeadDim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(k_cache);
    TensorMatcher({-1, kKVHeads * kHeadDim})
        .with_dtype(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(v_cache);
    TensorMatcher({tokens_})
        .with_dtype<int64_t, int32_t>(loc_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(out_loc);

    const auto tokens = static_cast<int>(tokens_.unwrap());
    if (tokens == 0) {
      return;
    }
    const auto device = device_.unwrap();
    const auto q_rows = tokens * kQHeads;
    const auto k_rows = tokens * kKVHeads;

    auto launch = [&](auto type_tag) {
      using DType = typename decltype(type_tag)::type;
      const QKNormRopeParams params{
          .positions = positions.data_ptr(),
          .q_ptr = q.data_ptr(),
          .k_ptr = k.data_ptr(),
          .q_weight_ptr = q_weight.data_ptr(),
          .k_weight_ptr = k_weight.data_ptr(),
          .cos_sin_cache = static_cast<const float*>(cos_sin_cache.data_ptr()),
          .q_stride_t = static_cast<int>(q.stride(0)),
          .k_stride_t = static_cast<int>(k.stride(0)),
          .cache_stride_t = static_cast<int>(cos_sin_cache.stride(0)),
          .q_rows = q_rows,
          .k_rows = k_rows,
          .positions_i64 = positions_dtype_.unwrap().bits == 64,
          .eps = static_cast<float>(eps),
          .k_cache = k_cache.data_ptr(),
          .v_cache = v_cache.data_ptr(),
          .v_ptr = v.data_ptr(),
          .out_loc = out_loc.data_ptr(),
          .kv_cache_stride_t = static_cast<long>(k_cache.stride(0)),
          .v_stride_t = static_cast<int>(v.stride(0)),
          .loc_i64 = loc_dtype_.unwrap().bits == 64,
          .store_kv = true,
      };
      const auto blocks = static_cast<unsigned>(
          (q_rows + 2 * k_rows + kBlockM - 1) / kBlockM);
      LaunchKernel(blocks, dim3(32, kBlockM, 1), device)(
          qk_norm_rope_kernel<DType, kQHeads, kKVHeads, kHeadDim, kBlockM>,
          params);
    };

    if (data_dtype_.unwrap().code == DLDataTypeCode::kDLBfloat) {
      launch(TypeTag<__nv_bfloat16>{});
    } else {
      launch(TypeTag<__half>{});
    }
  }
};

}  // namespace
