#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <type_traits>

namespace {

template <typename T>
__device__ __forceinline__ float scalar_to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float scalar_to_float<__nv_bfloat16>(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half v) {
  return __half2float(v);
}

// Vectorized load: VEC_SIZE bf16/fp16 elements at `src` (must be VEC_BYTES-aligned).
template <typename DType, int VEC_SIZE>
__device__ __forceinline__ void vec_load(DType* regs, const DType* src) {
  constexpr int VEC_BYTES = VEC_SIZE * sizeof(DType);
  static_assert(VEC_BYTES == 8 || VEC_BYTES == 16 || VEC_BYTES == 32);
  if constexpr (VEC_BYTES == 8) {
    *reinterpret_cast<uint2*>(regs) = *reinterpret_cast<const uint2*>(src);
  } else if constexpr (VEC_BYTES == 16) {
    *reinterpret_cast<uint4*>(regs) = *reinterpret_cast<const uint4*>(src);
  } else {
    uint4* d = reinterpret_cast<uint4*>(regs);
    const uint4* s = reinterpret_cast<const uint4*>(src);
    d[0] = s[0];
    d[1] = s[1];
  }
}

// Quantize VEC_SIZE fp32 values into FP8 e4m3 bytes via the CUDA FP8x2 PTX
// intrinsic and store contiguous uint32 chunks.
template <int VEC_SIZE>
__device__ __forceinline__ void vec_quant_store_fp8_from_floats(
    uint8_t* dst, const float* values, float inv_y) {
  static_assert(VEC_SIZE % 2 == 0);
  static_assert(VEC_SIZE % 4 == 0);
  __nv_fp8x2_storage_t out[VEC_SIZE / 2];
#pragma unroll
  for (int i = 0; i < VEC_SIZE; i += 2) {
    out[i / 2] = __nv_cvt_float2_to_fp8x2(
        make_float2(values[i] * inv_y, values[i + 1] * inv_y),
        __NV_SATFINITE, __NV_E4M3);
  }
  static_assert(sizeof(out) == VEC_SIZE);
  constexpr int N_UINT = VEC_SIZE / 4;
  uint32_t* d = reinterpret_cast<uint32_t*>(dst);
  const uint32_t* s = reinterpret_cast<const uint32_t*>(out);
#pragma unroll
  for (int i = 0; i < N_UINT; ++i) d[i] = s[i];
}

template <int THREADS_PER_GROUP>
__device__ __forceinline__ unsigned subgroup_mask(unsigned tid) {
  if constexpr (THREADS_PER_GROUP == 32) {
    return 0xffffffffu;
  } else {
    constexpr unsigned bits = (1u << THREADS_PER_GROUP) - 1u;
    const unsigned warp_lane = tid & 31u;
    return bits << (warp_lane & ~(THREADS_PER_GROUP - 1));
  }
}

__device__ __forceinline__ uint8_t ceil_to_ue8m0_exp_byte(float scale) {
  scale = fmaxf(scale, 1e-10f);
  const uint32_t bits = __float_as_uint(scale);
  return static_cast<uint8_t>(
      ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1u : 0u));
}

template <int block_count>
__device__ __forceinline__ void psum_token_bounds(
    int32_t n_tokens,
    int32_t worker_id,
    int32_t& lower,
    int32_t& upper) {
  if (n_tokens < block_count && worker_id < n_tokens) {
    lower = worker_id;
    upper = worker_id + 1;
    return;
  }
  const int32_t chunk_size = n_tokens / block_count;
  const int32_t residual = n_tokens - chunk_size * block_count;
  auto calc = [&](int32_t id) {
    if (id < residual) {
      return min(n_tokens, id * (chunk_size + 1));
    }
    return min(n_tokens, id * chunk_size + residual);
  };
  lower = calc(worker_id);
  upper = calc(worker_id + 1);
}

__device__ __forceinline__ int32_t psum_find_expert(
    const int32_t* __restrict__ compact_offsets,
    int32_t num_experts,
    int32_t compact_token) {
  int32_t lo = 0;
  int32_t hi = num_experts;
  while (lo + 1 < hi) {
    const int32_t mid = (lo + hi) >> 1;
    if (compact_offsets[mid] <= compact_token) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

template <typename DType, int block_count, int THREADS_PER_GROUP>
__global__ void psum_silu_mul_fp8_packed_compact_all_groups_kernel(
    const DType* __restrict__ x,
    uint8_t* __restrict__ y,
    uint32_t* __restrict__ s_packed,
    const int32_t* __restrict__ psum,
    const float* __restrict__ topk_weights,
    int groups_per_row,
    int m,
    int tma_aligned_m,
    int hidden,
    int num_experts,
    int alignment,
    bool apply_topk,
    float eps,
    float max_8bit) {
  constexpr int GROUP_SIZE = 128;
  constexpr int VEC_SIZE = GROUP_SIZE / THREADS_PER_GROUP;
  static_assert(GROUP_SIZE == THREADS_PER_GROUP * VEC_SIZE);
  static_assert(VEC_SIZE % 4 == 0);

  extern __shared__ int32_t shared_offsets[];
  int32_t* compact_offsets = shared_offsets;
  int32_t* physical_starts = shared_offsets + num_experts + 1;

  if (threadIdx.x == 0) {
    int32_t compact = 0;
    int prev_end = 0;
    compact_offsets[0] = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      const int end = psum[expert];
      const int start = ((prev_end + alignment - 1) / alignment) * alignment;
      const int count = end > start ? end - start : 0;
      physical_starts[expert] = start;
      compact += count;
      compact_offsets[expert + 1] = compact;
      prev_end = end;
    }
  }
  __syncthreads();

  const int32_t total_tokens = compact_offsets[num_experts];
  int32_t lower = 0;
  int32_t upper = 0;
  psum_token_bounds<block_count>(
      total_tokens, static_cast<int32_t>(blockIdx.x), lower, upper);
  if (lower >= upper) {
    return;
  }

  const int sf_k_idx = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;
  const unsigned sg_mask = subgroup_mask<THREADS_PER_GROUP>(threadIdx.x);
  const bool is_valid_sf_k = sf_k_idx < groups_per_row;
  const int padding_groups = ((groups_per_row + 3) / 4) * 4 - groups_per_row;

  int32_t expert = psum_find_expert(compact_offsets, num_experts, lower);
  for (int32_t compact_token = lower; compact_token < upper; ++compact_token) {
    while (expert + 1 < num_experts &&
           compact_token >= compact_offsets[expert + 1]) {
      ++expert;
    }
    const int row =
        physical_starts[expert] + compact_token - compact_offsets[expert];
    if (row >= m) {
      continue;
    }

    if (is_valid_sf_k) {
      alignas(16) DType gate_regs[VEC_SIZE];
      alignas(16) DType up_regs[VEC_SIZE];
      float values[VEC_SIZE];
      float local_absmax = eps;

      const int64_t row_base = static_cast<int64_t>(row) * (2 * hidden);
      const int col_lane = sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
      vec_load<DType, VEC_SIZE>(gate_regs, x + row_base + col_lane);
      vec_load<DType, VEC_SIZE>(up_regs, x + row_base + hidden + col_lane);
      const float tw = apply_topk ? topk_weights[row] : 1.0f;
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        const float g = scalar_to_float<DType>(gate_regs[i]);
        const float u = scalar_to_float<DType>(up_regs[i]);
        const float silu = g / (1.0f + __expf(-g));
        values[i] = silu * u * tw;
        local_absmax = fmaxf(local_absmax, fabsf(values[i]));
      }

#pragma unroll
      for (int off = THREADS_PER_GROUP / 2; off > 0; off >>= 1) {
        local_absmax = fmaxf(local_absmax,
                             __shfl_xor_sync(sg_mask, local_absmax, off));
      }

      const uint8_t exp_byte =
          ceil_to_ue8m0_exp_byte(local_absmax / max_8bit);
      if (lane_id == 0) {
        const int sf_k_pack_idx = sf_k_idx / 4;
        const int pos = sf_k_idx % 4;
        const int64_t out_idx =
            static_cast<int64_t>(sf_k_pack_idx) * tma_aligned_m + row;
        reinterpret_cast<uint8_t*>(s_packed)[out_idx * 4 + pos] = exp_byte;
      }

      const float scale =
          __uint_as_float(static_cast<uint32_t>(exp_byte) << 23);
      const float inv_y = 1.0f / scale;
      uint8_t* group_output =
          y + static_cast<int64_t>(row) * hidden + col_lane;
      vec_quant_store_fp8_from_floats<VEC_SIZE>(group_output, values, inv_y);
    }

    if (threadIdx.x < padding_groups) {
      const int sf_k_pad = groups_per_row + threadIdx.x;
      const int sf_k_pack_idx = sf_k_pad / 4;
      const int pos = sf_k_pad % 4;
      const int64_t out_idx =
          static_cast<int64_t>(sf_k_pack_idx) * tma_aligned_m + row;
      reinterpret_cast<uint8_t*>(s_packed)[out_idx * 4 + pos] = 0;
    }
  }
}

template <typename T>
struct TypeTag {
  using type = T;
};

struct PsumSiluMulFp8PackedKernel {
  static void run(
      const tvm::ffi::TensorView x,                   // (m, 2*h) bf16/fp16
      const tvm::ffi::TensorView y_u8,                // (m, h) uint8
      const tvm::ffi::TensorView s_i32,               // (m, packed_groups) int32, strides (1, tma_aligned_m)
      const tvm::ffi::TensorView psum,                // (num_experts,) int32 cumulative
      const tvm::ffi::TensorView topk_weights,        // (m,) float32 or (1,) when unused
      int64_t alignment,
      bool apply_topk) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto m_ = SymbolicSize{"M"};
    auto h2_ = SymbolicSize{"H2"};
    auto h_ = SymbolicSize{"H"};
    auto pg_ = SymbolicSize{"PG"};
    auto e_ = SymbolicSize{"E"};

    TensorMatcher({m_, h2_})
        .with_dtype<__nv_bfloat16, __half>(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(x);
    TensorMatcher({m_, h_})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(y_u8);
    TensorMatcher({m_, pg_})
        .with_strides({-1, -1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(s_i32);
    TensorMatcher({e_})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(psum);
    TensorMatcher({-1})
        .with_dtype<float>()
        .with_device<kDLCUDA>(device_)
        .verify(topk_weights);

    constexpr int GROUP_SIZE = 128;
    const auto m = m_.unwrap();
    const auto h = h_.unwrap();
    const auto h2 = h2_.unwrap();
    const auto pg = pg_.unwrap();
    const auto num_experts = e_.unwrap();
    RuntimeCheck(h2 == 2 * h, "expected x.shape[1] == 2 * y.shape[1], got ",
                 h2, " vs 2*", h);
    RuntimeCheck(h % GROUP_SIZE == 0, "h=", h,
                 " must be divisible by GROUP_SIZE=", GROUP_SIZE);
    const auto groups_per_row = h / GROUP_SIZE;
    const auto k_num_packed_sfk = (groups_per_row + 3) / 4;
    const auto tma_aligned_m = ((m + 3) / 4) * 4;
    RuntimeCheck(pg == k_num_packed_sfk,
                 "packed_groups=", pg, " expected ", k_num_packed_sfk);
    RuntimeCheck(s_i32.stride(0) == 1 && s_i32.stride(1) == tma_aligned_m,
                 "s_i32 strides must be (1, tma_aligned_m=", tma_aligned_m,
                 "), got (", s_i32.stride(0), ", ", s_i32.stride(1), ")");
    if (m == 0 || h == 0) {
      return;
    }
    const auto device = device_.unwrap();
    RuntimeCheck(groups_per_row > 0 && groups_per_row * 8 <= 1024,
                 "compact_all psum_silu_mul_fp8_packed requires ",
                 "0 < groups_per_row * 8 <= 1024, got groups_per_row=",
                 groups_per_row);
    if (const char* mode = std::getenv("MINISGL_PSUM_SILU_QUANT_MODE")) {
      RuntimeCheck(std::strcmp(mode, "auto") == 0 ||
                   std::strcmp(mode, "compact_all") == 0 ||
                   std::strcmp(mode, "compact_all_groups") == 0,
                   "MINISGL_PSUM_SILU_QUANT_MODE must be auto or compact_all");
    }

    auto launch_compact_all_groups = [&](auto type_tag, auto block_count_tag) {
      using DType = typename decltype(type_tag)::type;
      constexpr int BLOCK_COUNT = decltype(block_count_tag)::value;
      RuntimeCheck(groups_per_row > 0 && groups_per_row * 8 <= 1024,
                   "invalid compact_all_groups psum_silu_mul_fp8_packed "
                   "block size: ",
                   groups_per_row * 8);
      const size_t shared_mem =
          static_cast<size_t>(2 * num_experts + 1) * sizeof(int32_t);
      LaunchKernel(dim3(static_cast<unsigned>(BLOCK_COUNT)),
                   dim3(static_cast<unsigned>(groups_per_row * 8)), device,
                   shared_mem)(
          psum_silu_mul_fp8_packed_compact_all_groups_kernel<DType, BLOCK_COUNT, 8>,
          static_cast<const DType*>(x.data_ptr()),
          static_cast<uint8_t*>(y_u8.data_ptr()),
          reinterpret_cast<uint32_t*>(s_i32.data_ptr()),
          static_cast<const int32_t*>(psum.data_ptr()),
          static_cast<const float*>(topk_weights.data_ptr()),
          static_cast<int>(groups_per_row),
          static_cast<int>(m),
          static_cast<int>(tma_aligned_m),
          static_cast<int>(h),
          static_cast<int>(num_experts),
          static_cast<int>(alignment),
          apply_topk,
          1e-10f,
          448.0f);
    };

    auto dispatch = [&](auto type_tag) {
      launch_compact_all_groups(
          type_tag, std::integral_constant<int, 132 * 32>{});
    };

    if (data_dtype_.unwrap().code == DLDataTypeCode::kDLBfloat) {
      dispatch(TypeTag<__nv_bfloat16>{});
    } else {
      dispatch(TypeTag<__half>{});
    }
  }
};

// Compatibility entry for callers that still pass manual_config. It intentionally
// ignores the manual tiling knobs so the psum path compiles one CUDA kernel only.
struct PsumSiluMulFp8PackedKernelManual {
  static void run(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView y_u8,
      const tvm::ffi::TensorView s_i32,
      const tvm::ffi::TensorView psum,
      const tvm::ffi::TensorView topk_weights,
      int64_t alignment,
      bool apply_topk,
      int64_t threads_per_group,
      int64_t groups_per_block_x,
      int64_t rows_per_block) {
    (void)threads_per_group;
    (void)groups_per_block_x;
    (void)rows_per_block;
    PsumSiluMulFp8PackedKernel::run(
        x, y_u8, s_i32, psum, topk_weights, alignment, apply_topk);
  }
};

}  // namespace
