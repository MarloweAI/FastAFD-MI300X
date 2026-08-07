#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

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

// Vectorized load: VEC_SIZE bf16/fp16 elements at `src` (must be VEC_BYTES-aligned)
// into `regs`. Uses uint2 / uint4 / 2x-uint4 depending on VEC_BYTES.
template <typename DType, int VEC_SIZE>
__device__ __forceinline__ void vec_load(DType* regs, const DType* src) {
  constexpr int VEC_BYTES = VEC_SIZE * sizeof(DType);
  static_assert(VEC_BYTES == 8 || VEC_BYTES == 16 || VEC_BYTES == 32,
                "unsupported VEC_BYTES");
  if constexpr (VEC_BYTES == 8) {
    *reinterpret_cast<uint2*>(regs) = *reinterpret_cast<const uint2*>(src);
  } else if constexpr (VEC_BYTES == 16) {
    *reinterpret_cast<uint4*>(regs) = *reinterpret_cast<const uint4*>(src);
  } else {  // 32
    uint4* d = reinterpret_cast<uint4*>(regs);
    const uint4* s = reinterpret_cast<const uint4*>(src);
    d[0] = s[0];
    d[1] = s[1];
  }
}

// Quantize VEC_SIZE float elements into FP8 e4m3 bytes and store contiguous at
// `dst`. Uses sglang/DeepEP-style `__nv_cvt_float2_to_fp8x2(_, __NV_SATFINITE,
// __NV_E4M3)` PTX intrinsic (1 instruction per 2 elements + built-in saturate)
// instead of per-element __nv_fp8_e4m3 constructor + explicit fminf/fmaxf.
template <typename DType, int VEC_SIZE>
__device__ __forceinline__ void vec_quant_store_fp8(
    uint8_t* dst, const DType* regs, float inv_y, float /*min_8 unused*/, float /*max_8 unused*/) {
  static_assert(VEC_SIZE % 2 == 0, "VEC_SIZE must be even");
  static_assert(VEC_SIZE % 4 == 0, "VEC_SIZE must be a multiple of 4");
  __nv_fp8x2_storage_t out[VEC_SIZE / 2];
#pragma unroll
  for (int i = 0; i < VEC_SIZE; i += 2) {
    const float a = scalar_to_float<DType>(regs[i]) * inv_y;
    const float b = scalar_to_float<DType>(regs[i + 1]) * inv_y;
    out[i / 2] = __nv_cvt_float2_to_fp8x2(make_float2(a, b),
                                          __NV_SATFINITE, __NV_E4M3);
  }
  static_assert(sizeof(out) == VEC_SIZE);
  constexpr int N_UINT = VEC_SIZE / 4;
  uint32_t* d = reinterpret_cast<uint32_t*>(dst);
  const uint32_t* s = reinterpret_cast<const uint32_t*>(out);
#pragma unroll
  for (int i = 0; i < N_UINT; ++i) d[i] = s[i];
}

// Subgroup mask for shfl_xor across THREADS_PER_GROUP lanes within a warp.
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

// ============================================================================
// Unified per-token FP8 packed UE8M0 quantization kernel.
//
// Templated on three axes for SM-occupancy autotuning:
//   THREADS_PER_GROUP   : 8, 16, or 32 (covers vLLM register-resident + the
//                         minisgl v1 "warp-per-group" designs)
//   GROUPS_PER_BLOCK_X  : groups packed along K axis per block (1, 2, 4, 8, 16)
//   ROWS_PER_BLOCK      : MN rows packed per block (1, 2, 4, 8, 16)
//
// Block size = THREADS_PER_GROUP * GROUPS_PER_BLOCK_X * ROWS_PER_BLOCK.
// Each warp lane owns VEC_SIZE = 128/THREADS_PER_GROUP elements of its group;
// load + reduce + UE8M0 scale + quant + pack stay register-resident.
//
// Output scale layout: stride=(1, tma_aligned_mn) int32, column-major aligned
// (matches DeepGEMM/vLLM TMA-friendly packed-scale layout).
// ============================================================================
template <typename DType, int THREADS_PER_GROUP, int GROUPS_PER_BLOCK_X, int ROWS_PER_BLOCK>
__global__ void per_token_cast_fp8_packed_kernel(
    const DType* __restrict__ input,
    uint8_t* __restrict__ output_q,
    uint32_t* __restrict__ output_s_packed,
    int padded_groups_per_row,
    int groups_per_row,
    int mn,
    int output_q_mn_extent,
    int tma_aligned_mn,
    int64_t num_scale_elems,
    float eps,
    float min_8bit,
    float max_8bit) {
  constexpr int GROUP_SIZE = 128;
  constexpr int VEC_SIZE = GROUP_SIZE / THREADS_PER_GROUP;
  static_assert(GROUP_SIZE == THREADS_PER_GROUP * VEC_SIZE);
  static_assert(VEC_SIZE % 4 == 0);
  static_assert(VEC_SIZE * sizeof(DType) % 8 == 0);

  const int local_group_id = threadIdx.x / THREADS_PER_GROUP;
  const int lane_id = threadIdx.x % THREADS_PER_GROUP;

  const int sf_k_local = local_group_id % GROUPS_PER_BLOCK_X;
  const int row_local = local_group_id / GROUPS_PER_BLOCK_X;
  const int sf_k_idx = blockIdx.x * GROUPS_PER_BLOCK_X + sf_k_local;
  const int mn_idx = blockIdx.y * ROWS_PER_BLOCK + row_local;

  if (mn_idx >= tma_aligned_mn) {
    return;
  }
  const bool is_valid_group = (mn_idx < mn) && (sf_k_idx < groups_per_row);

  alignas(16) DType regs[VEC_SIZE];
  float local_absmax = eps;
  if (is_valid_group) {
    const DType* group_input =
        input + static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
        sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
    vec_load<DType, VEC_SIZE>(regs, group_input);
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float v = fabsf(scalar_to_float<DType>(regs[i]));
      local_absmax = fmaxf(local_absmax, v);
    }
  }

  const unsigned mask = subgroup_mask<THREADS_PER_GROUP>(threadIdx.x);
#pragma unroll
  for (int off = THREADS_PER_GROUP / 2; off > 0; off >>= 1) {
    local_absmax = fmaxf(local_absmax, __shfl_xor_sync(mask, local_absmax, off));
  }

  float y_s = local_absmax / max_8bit;
  y_s = fmaxf(y_s, 1e-10f);
  uint32_t bits = __float_as_uint(y_s);
  uint8_t exp_byte = static_cast<uint8_t>(
      ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1u : 0u));

  if (lane_id == 0) {
    const int sf_k_pack_idx = sf_k_idx / 4;
    const int pos = sf_k_idx % 4;
    const int64_t out_idx =
        static_cast<int64_t>(sf_k_pack_idx) * tma_aligned_mn + mn_idx;
    if (is_valid_group) {
      reinterpret_cast<uint8_t*>(output_s_packed)[out_idx * 4 + pos] = exp_byte;
    } else if (out_idx < num_scale_elems) {
      reinterpret_cast<uint8_t*>(output_s_packed)[out_idx * 4 + pos] = 0;
    }
  }

  if (!is_valid_group) {
    if (sf_k_idx < groups_per_row && mn_idx >= mn &&
        mn_idx < output_q_mn_extent) {
      uint8_t* group_output =
          output_q +
          static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
          sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
      // Zero-fill VEC_SIZE bytes via memset (vectorized uint chunks).
      constexpr int N_UINT = VEC_SIZE / 4;
      uint32_t* d = reinterpret_cast<uint32_t*>(group_output);
#pragma unroll
      for (int i = 0; i < N_UINT; ++i) d[i] = 0u;
    }
    return;
  }

  float y_s_q = __uint_as_float(static_cast<uint32_t>(exp_byte) << 23);
  float inv_y = 1.0f / y_s_q;
  uint8_t* group_output =
      output_q +
      static_cast<int64_t>(mn_idx) * groups_per_row * GROUP_SIZE +
      sf_k_idx * GROUP_SIZE + lane_id * VEC_SIZE;
  vec_quant_store_fp8<DType, VEC_SIZE>(group_output, regs, inv_y, min_8bit, max_8bit);
}

template <typename T>
struct TypeTag {
  using type = T;
};

struct LaunchConfig {
  int threads_per_group;
  int groups_per_block_x;
  int rows_per_block;
};

constexpr int64_t kMaxGridDimY = 65535;

inline int64_t div_up_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

// Pick the most-utilizing block config based on grid count.
// The autotuning heuristic is derived from NSYS sweep on GB200 (see
// docs/qwen3_*_fp8_kernel_profile.md or bench output):
//   - At very small (m, n), config (32, 4, 1) ("v1 style") gives the most
//     blocks (~m * packed_groups), best SM utilization.
//   - For larger problems, register-resident (8, 16/8/4, 1/2/4) ("v2 style")
//     wins by 1.5-2x via wider per-thread LDG (2x uint4 = 32 B).
inline LaunchConfig pick_config(int64_t mn, int64_t padded_groups_per_row) {
  // Heuristic: empirically derived on GB200 (NSYS sweep over shapes).
  // At very small (m, n), config (32, 4, 1) ("v1 style") wins because the
  // larger thread count (32 vs 8 per group) keeps SMs busier when there are
  // too few v2 blocks; the per-thread work is smaller (4 vs 16 elems) but the
  // launch fan-out is what matters at low block count.
  // Once v2 has ~200+ blocks (GB200 has 144 SMs), the wider per-thread LDG
  // (2x uint4 = 32 B) pays off and (8, *, *) wins by up to 2× at large m.
  constexpr int64_t kBlocksThreshold = 200;

  auto v2_kx = [](int64_t padded) -> int {
    if (padded % 16 == 0) return 16;
    if (padded % 8 == 0) return 8;
    return 4;
  };

  int kx = v2_kx(padded_groups_per_row);
  int ry = 16 / kx;
  while (div_up_i64(mn, ry) > kMaxGridDimY && kx > 1) {
    kx /= 2;
    ry = 16 / kx;
  }
  host::RuntimeCheck(
      div_up_i64(mn, ry) <= kMaxGridDimY,
      "per_token_fp8_packed launch would exceed CUDA grid.y: rows=",
      mn, " rows_per_block=", ry, " grid.y=", div_up_i64(mn, ry),
      " max=", kMaxGridDimY);
  const int64_t v2_blocks =
      (padded_groups_per_row / kx) * div_up_i64(mn, ry);
  if (v2_blocks >= kBlocksThreshold) {
    return {8, kx, ry};
  }
  if (div_up_i64(mn, 1) > kMaxGridDimY) {
    return {8, kx, ry};
  }
  // Fall back to v1-style (32 threads/group, 4 groups/block, 1 row/block):
  // generates more blocks at small (m, n), better SM occupancy.
  return {32, 4, 1};
}

// Public class — single canonical packed-UE8M0 FP8 quant entry. Auto-dispatches
// across (THREADS_PER_GROUP, GROUPS_PER_BLOCK_X, ROWS_PER_BLOCK) instantiations
// based on (m, n) for best SM utilization.
struct PerTokenCastFp8PackedKernel {
  static void run(
      const tvm::ffi::TensorView x,       // (m, n) bf16 / fp16
      const tvm::ffi::TensorView y_u8,    // (m, n) uint8 (fp8 bytes)
      const tvm::ffi::TensorView s_i32) { // (m, packed_groups) int32, strides (1, tma_aligned_mn)
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto m_ = SymbolicSize{"M"};
    auto n_ = SymbolicSize{"N"};
    auto pg_ = SymbolicSize{"PG"};

    TensorMatcher({m_, n_})
        .with_dtype<__nv_bfloat16, __half>(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(x);
    TensorMatcher({m_, n_})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(y_u8);
    TensorMatcher({m_, pg_})
        .with_strides({-1, -1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(s_i32);

    constexpr int GROUP_SIZE = 128;
    const auto m = m_.unwrap();
    const auto n = n_.unwrap();
    const auto pg = pg_.unwrap();
    RuntimeCheck(n % GROUP_SIZE == 0, "n=", n,
                 " must be divisible by GROUP_SIZE=", GROUP_SIZE);
    const auto groups_per_row = n / GROUP_SIZE;
    const auto k_num_packed_sfk = (groups_per_row + 3) / 4;
    const auto tma_aligned_mn = ((m + 3) / 4) * 4;
    RuntimeCheck(pg == k_num_packed_sfk,
                 "packed_groups=", pg, " expected ", k_num_packed_sfk);
    RuntimeCheck(s_i32.stride(0) == 1 &&
                     s_i32.stride(1) == tma_aligned_mn,
                 "s_i32 strides must be (1, tma_aligned_mn=", tma_aligned_mn,
                 "), got (", s_i32.stride(0), ", ", s_i32.stride(1), ")");
    if (m == 0 || n == 0) {
      return;
    }
    const auto device = device_.unwrap();
    const auto padded_groups_per_row = k_num_packed_sfk * 4;
    const auto num_scale_elems =
        m + (k_num_packed_sfk - 1) * tma_aligned_mn;
    const LaunchConfig cfg = pick_config(tma_aligned_mn, padded_groups_per_row);

    auto launch = [&](auto type_tag, auto tpg_tag, auto kx_tag, auto ry_tag) {
      using DType = typename decltype(type_tag)::type;
      constexpr int TPG = decltype(tpg_tag)::value;
      constexpr int KX = decltype(kx_tag)::value;
      constexpr int RY = decltype(ry_tag)::value;
      const auto blocks_x = padded_groups_per_row / KX;
      const auto blocks_y = (tma_aligned_mn + RY - 1) / RY;
      RuntimeCheck(blocks_x > 0 && blocks_y > 0,
                   "invalid per_token_fp8_packed grid dims (",
                   blocks_x, ", ", blocks_y, ")");
      RuntimeCheck(blocks_y <= kMaxGridDimY,
                   "per_token_fp8_packed grid.y=", blocks_y,
                   " exceeds CUDA limit ", kMaxGridDimY,
                   " for rows=", tma_aligned_mn,
                   " rows_per_block=", RY);
      const dim3 grid(static_cast<unsigned>(blocks_x),
                      static_cast<unsigned>(blocks_y));
      const dim3 block(static_cast<unsigned>(TPG * KX * RY));
      LaunchKernel(grid, block, device)(
          per_token_cast_fp8_packed_kernel<DType, TPG, KX, RY>,
          static_cast<const DType*>(x.data_ptr()),
          static_cast<uint8_t*>(y_u8.data_ptr()),
          reinterpret_cast<uint32_t*>(s_i32.data_ptr()),
          static_cast<int>(padded_groups_per_row),
          static_cast<int>(groups_per_row),
          static_cast<int>(m),
          static_cast<int>(m),
          static_cast<int>(tma_aligned_mn),
          static_cast<int64_t>(num_scale_elems),
          1e-10f,
          -448.0f,
          448.0f);
    };

    auto pick_kx_ry = [&](auto type_tag, auto tpg_tag) {
      if (cfg.groups_per_block_x == 16 && cfg.rows_per_block == 1) {
        launch(type_tag, tpg_tag,
               std::integral_constant<int, 16>{},
               std::integral_constant<int, 1>{});
      } else if (cfg.groups_per_block_x == 8 && cfg.rows_per_block == 2) {
        launch(type_tag, tpg_tag,
               std::integral_constant<int, 8>{},
               std::integral_constant<int, 2>{});
      } else if (cfg.groups_per_block_x == 4 && cfg.rows_per_block == 4) {
        launch(type_tag, tpg_tag,
               std::integral_constant<int, 4>{},
               std::integral_constant<int, 4>{});
      } else if (cfg.groups_per_block_x == 2 && cfg.rows_per_block == 8) {
        launch(type_tag, tpg_tag,
               std::integral_constant<int, 2>{},
               std::integral_constant<int, 8>{});
      } else if (cfg.groups_per_block_x == 1 && cfg.rows_per_block == 16) {
        launch(type_tag, tpg_tag,
               std::integral_constant<int, 1>{},
               std::integral_constant<int, 16>{});
      } else if (cfg.groups_per_block_x == 4 && cfg.rows_per_block == 1) {
        launch(type_tag, tpg_tag,
               std::integral_constant<int, 4>{},
               std::integral_constant<int, 1>{});
      } else {
        RuntimeCheck(false, "unsupported (gpb_x, rpb)=(",
                     cfg.groups_per_block_x, ", ", cfg.rows_per_block, ")");
      }
    };

    auto dispatch = [&](auto type_tag) {
      if (cfg.threads_per_group == 8) {
        pick_kx_ry(type_tag, std::integral_constant<int, 8>{});
      } else if (cfg.threads_per_group == 32) {
        pick_kx_ry(type_tag, std::integral_constant<int, 32>{});
      } else {
        RuntimeCheck(false, "unsupported THREADS_PER_GROUP=",
                     cfg.threads_per_group);
      }
    };

    if (data_dtype_.unwrap().code == DLDataTypeCode::kDLBfloat) {
      dispatch(TypeTag<__nv_bfloat16>{});
    } else {
      dispatch(TypeTag<__half>{});
    }
  }
};

// Manual-config entry used by the bench/test script to compare individual
// (TPG, KX, RY) variants without the auto-dispatcher. The dispatcher above is
// what production code goes through.
struct PerTokenCastFp8PackedKernelManual {
  static void run(
      const tvm::ffi::TensorView x,
      const tvm::ffi::TensorView y_u8,
      const tvm::ffi::TensorView s_i32,
      int64_t threads_per_group,
      int64_t groups_per_block_x,
      int64_t rows_per_block) {
    using namespace host;
    auto device_ = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};
    auto m_ = SymbolicSize{"M"};
    auto n_ = SymbolicSize{"N"};
    auto pg_ = SymbolicSize{"PG"};

    TensorMatcher({m_, n_})
        .with_dtype<__nv_bfloat16, __half>(data_dtype_)
        .with_device<kDLCUDA>(device_)
        .verify(x);
    TensorMatcher({m_, n_})
        .with_dtype<uint8_t>()
        .with_device<kDLCUDA>(device_)
        .verify(y_u8);
    TensorMatcher({m_, pg_})
        .with_strides({-1, -1})
        .with_dtype<int32_t>()
        .with_device<kDLCUDA>(device_)
        .verify(s_i32);

    constexpr int GROUP_SIZE = 128;
    const auto m = m_.unwrap();
    const auto n = n_.unwrap();
    const auto groups_per_row = n / GROUP_SIZE;
    const auto k_num_packed_sfk = (groups_per_row + 3) / 4;
    const auto tma_aligned_mn = ((m + 3) / 4) * 4;
    if (m == 0 || n == 0) return;
    const auto device = device_.unwrap();
    const auto padded_groups_per_row = k_num_packed_sfk * 4;
    const auto num_scale_elems =
        m + (k_num_packed_sfk - 1) * tma_aligned_mn;

    auto launch = [&](auto type_tag, auto tpg_tag, auto kx_tag, auto ry_tag) {
      using DType = typename decltype(type_tag)::type;
      constexpr int TPG = decltype(tpg_tag)::value;
      constexpr int KX = decltype(kx_tag)::value;
      constexpr int RY = decltype(ry_tag)::value;
      const auto blocks_x = padded_groups_per_row / KX;
      const auto blocks_y = (tma_aligned_mn + RY - 1) / RY;
      RuntimeCheck(blocks_x > 0 && blocks_y > 0,
                   "invalid manual per_token_fp8_packed grid dims (",
                   blocks_x, ", ", blocks_y, ")");
      RuntimeCheck(blocks_y <= kMaxGridDimY,
                   "manual per_token_fp8_packed grid.y=", blocks_y,
                   " exceeds CUDA limit ", kMaxGridDimY,
                   " for rows=", tma_aligned_mn,
                   " rows_per_block=", RY);
      const dim3 grid(static_cast<unsigned>(blocks_x),
                      static_cast<unsigned>(blocks_y));
      const dim3 block(static_cast<unsigned>(TPG * KX * RY));
      LaunchKernel(grid, block, device)(
          per_token_cast_fp8_packed_kernel<DType, TPG, KX, RY>,
          static_cast<const DType*>(x.data_ptr()),
          static_cast<uint8_t*>(y_u8.data_ptr()),
          reinterpret_cast<uint32_t*>(s_i32.data_ptr()),
          static_cast<int>(padded_groups_per_row),
          static_cast<int>(groups_per_row),
          static_cast<int>(m), static_cast<int>(m),
          static_cast<int>(tma_aligned_mn),
          static_cast<int64_t>(num_scale_elems),
          1e-10f, -448.0f, 448.0f);
    };

    auto choose_kx_ry = [&](auto type_tag, auto tpg_tag) {
      if (groups_per_block_x == 16 && rows_per_block == 1) {
        launch(type_tag, tpg_tag, std::integral_constant<int, 16>{}, std::integral_constant<int, 1>{});
      } else if (groups_per_block_x == 8 && rows_per_block == 2) {
        launch(type_tag, tpg_tag, std::integral_constant<int, 8>{}, std::integral_constant<int, 2>{});
      } else if (groups_per_block_x == 8 && rows_per_block == 1) {
        launch(type_tag, tpg_tag, std::integral_constant<int, 8>{}, std::integral_constant<int, 1>{});
      } else if (groups_per_block_x == 4 && rows_per_block == 4) {
        launch(type_tag, tpg_tag, std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
      } else if (groups_per_block_x == 2 && rows_per_block == 8) {
        launch(type_tag, tpg_tag, std::integral_constant<int, 2>{}, std::integral_constant<int, 8>{});
      } else if (groups_per_block_x == 1 && rows_per_block == 16) {
        launch(type_tag, tpg_tag, std::integral_constant<int, 1>{}, std::integral_constant<int, 16>{});
      } else if (groups_per_block_x == 4 && rows_per_block == 1) {
        launch(type_tag, tpg_tag, std::integral_constant<int, 4>{}, std::integral_constant<int, 1>{});
      } else {
        RuntimeCheck(false, "manual: unsupported (gpb_x, rpb)=(",
                     groups_per_block_x, ", ", rows_per_block, ")");
      }
    };

    auto dispatch = [&](auto type_tag) {
      if (threads_per_group == 8) {
        choose_kx_ry(type_tag, std::integral_constant<int, 8>{});
      } else if (threads_per_group == 16) {
        choose_kx_ry(type_tag, std::integral_constant<int, 16>{});
      } else if (threads_per_group == 32) {
        choose_kx_ry(type_tag, std::integral_constant<int, 32>{});
      } else {
        RuntimeCheck(false, "manual: unsupported THREADS_PER_GROUP=",
                     threads_per_group);
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
