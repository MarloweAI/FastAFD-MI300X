#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace {

inline constexpr int kWarpSize = 32;
inline constexpr int kMaxExperts = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

template <int kWarpsPerBlock>
__launch_bounds__(kWarpSize * kWarpsPerBlock) __global__
void minimax_route_fp32_logits_tile_kernel(
    const __nv_bfloat16* __restrict__ hidden,  // [M, H]
    const float* __restrict__ gate_w,          // [E, H]
    float* __restrict__ logits,                // [M, E]
    int num_tokens,
    int hidden_size,
    int num_experts) {
  const int token = blockIdx.x;
  const int expert_tile = blockIdx.y * kWarpsPerBlock;
  if (token >= num_tokens) return;
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int expert = expert_tile + warp;
  if (expert >= num_experts) return;

  const __nv_bfloat16* h_row = hidden + static_cast<size_t>(token) * hidden_size;
  const float* w_row = gate_w + static_cast<size_t>(expert) * hidden_size;
  float acc = 0.0f;
  for (int k = lane; k < hidden_size; k += kWarpSize) {
    acc = fmaf(__bfloat162float(h_row[k]), w_row[k], acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    logits[static_cast<size_t>(token) * num_experts + expert] = acc;
  }
}

template <int kWarpsPerBlock, bool kApplySigmoid = false>
__launch_bounds__(kWarpSize * kWarpsPerBlock) __global__
void minimax_route_fp32_logits_tile_shared_kernel(
    const __nv_bfloat16* __restrict__ hidden,  // [M, H]
    const float* __restrict__ gate_w,          // [E, H]
    float* __restrict__ logits,                // [M, E]
    uint8_t* __restrict__ x_q,                 // optional [M, H] e4m3
    uint8_t* __restrict__ x_sf,                // optional raw [M, H/32] ue8m0
    int num_tokens,
    int hidden_size,
    int num_experts) {
  extern __shared__ __nv_bfloat16 h_s[];
  const int token = blockIdx.x;
  const int expert_tile = blockIdx.y * kWarpsPerBlock;
  if (token >= num_tokens) return;
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;
  const int expert = expert_tile + warp;

  const __nv_bfloat16* h_row = hidden + static_cast<size_t>(token) * hidden_size;
  if ((hidden_size & 7) == 0) {
    const int vecs = hidden_size / 8;
    const auto h_vec = reinterpret_cast<const uint4*>(h_row);
    auto s_vec = reinterpret_cast<uint4*>(h_s);
    for (int v = tid; v < vecs; v += blockDim.x) {
      s_vec[v] = __ldg(h_vec + v);
    }
  } else {
    for (int k = tid; k < hidden_size; k += blockDim.x) {
      h_s[k] = h_row[k];
    }
  }
  __syncthreads();

  if (x_q != nullptr && expert_tile == 0) {
    const int num_groups = hidden_size / 32;
    for (int g = tid; g < num_groups; g += blockDim.x) {
      float vals[32];
      const auto src_vec = reinterpret_cast<const uint4*>(h_s + g * 32);
      uint4 raw[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        raw[j] = src_vec[j];
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const auto h = reinterpret_cast<const __nv_bfloat16*>(raw + j);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          vals[j * 8 + k] = __bfloat162float(h[k]);
        }
      }
      float amax = 1e-4f;
#pragma unroll
      for (int j = 0; j < 32; ++j) {
        amax = fmaxf(amax, fabsf(vals[j]));
      }
      const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
      const uint32_t e =
          ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) ? 1u : 0u);
      x_sf[static_cast<size_t>(token) * num_groups + g] =
          static_cast<uint8_t>(e);
      const float inv_scale = __uint_as_float((254u - e) << 23);
      uint32_t packed[8];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        reinterpret_cast<__nv_fp8x2_storage_t*>(packed)[j] =
            __nv_cvt_float2_to_fp8x2(
                make_float2(vals[2 * j] * inv_scale,
                            vals[2 * j + 1] * inv_scale),
                __NV_SATFINITE, __NV_E4M3);
      }
      auto dst = reinterpret_cast<uint4*>(
          x_q + static_cast<size_t>(token) * hidden_size + g * 32);
      dst[0] = *reinterpret_cast<uint4*>(packed);
      dst[1] = *reinterpret_cast<uint4*>(packed + 4);
    }
  }

  if (expert >= num_experts) return;

  const float* w_row = gate_w + static_cast<size_t>(expert) * hidden_size;
  float acc = 0.0f;
  int k = lane;
  for (; k + kWarpSize * 7 < hidden_size; k += kWarpSize * 8) {
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int idx = k + u * kWarpSize;
      acc = fmaf(__bfloat162float(h_s[idx]), w_row[idx], acc);
    }
  }
  for (; k < hidden_size; k += kWarpSize) {
    acc = fmaf(__bfloat162float(h_s[k]), w_row[k], acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    if constexpr (kApplySigmoid) {
      acc = __frcp_rn(1.0f + __expf(-acc));
    }
    logits[static_cast<size_t>(token) * num_experts + expert] = acc;
  }
}

}  // namespace

struct MiniMaxRouteFusedKernel {
  static void run_logits(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight) {
    using namespace host;
    auto num_tokens = SymbolicSize{"M"};
    auto hidden_size = SymbolicSize{"H"};
    auto num_experts = SymbolicSize{"E"};
    auto device_ref = SymbolicDevice{};
    auto hidden_dtype = SymbolicDType{};

    TensorMatcher({num_tokens, num_experts})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(logits);
    TensorMatcher({num_tokens, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(hidden_dtype)
        .verify(hidden);
    {
      const auto dt = hidden_dtype.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16,
                   "MiniMax route logits hidden must be bf16");
    }
    TensorMatcher({num_experts, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(gate_weight);

    const auto rows = static_cast<int>(num_tokens.unwrap());
    const auto experts = static_cast<int>(num_experts.unwrap());
    const auto hsize = static_cast<int>(hidden_size.unwrap());
    if (rows == 0) return;
    RuntimeCheck(experts > 0 && experts <= kMaxExperts,
                 "MiniMax route logits supports 1..256 experts");

    const auto stream = LaunchKernel::resolve_device(device_ref.unwrap());
    constexpr int kWarps = 8;
    const dim3 grid_dot(rows, (experts + kWarps - 1) / kWarps);
    minimax_route_fp32_logits_tile_shared_kernel<kWarps><<<
        grid_dot, kWarpSize * kWarps, hsize * sizeof(__nv_bfloat16), stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data_ptr()),
        static_cast<const float*>(gate_weight.data_ptr()),
        static_cast<float*>(logits.data_ptr()),
        nullptr, nullptr,
        rows, hsize, experts);
  }

  static void run_logits4(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight) {
    run_logits_impl<4>(
        logits,
        hidden,
        gate_weight,
        tvm::ffi::Optional<tvm::ffi::TensorView>(),
        tvm::ffi::Optional<tvm::ffi::TensorView>());
  }

  static void run_logits16(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight) {
    run_logits_impl<16>(
        logits,
        hidden,
        gate_weight,
        tvm::ffi::Optional<tvm::ffi::TensorView>(),
        tvm::ffi::Optional<tvm::ffi::TensorView>());
  }

  static void run_logits4_quant(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight,
      const tvm::ffi::TensorView x_q,
      const tvm::ffi::TensorView x_sf) {
    run_logits_impl<4>(logits, hidden, gate_weight, x_q, x_sf);
  }

  static void run_logits16_quant(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight,
      const tvm::ffi::TensorView x_q,
      const tvm::ffi::TensorView x_sf) {
    run_logits_impl<16>(logits, hidden, gate_weight, x_q, x_sf);
  }

  static void run_logits16_quant_sigmoid(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight,
      const tvm::ffi::TensorView x_q,
      const tvm::ffi::TensorView x_sf) {
    run_logits_impl<16, true>(logits, hidden, gate_weight, x_q, x_sf);
  }

 private:
  template <int kWarps, bool kApplySigmoid = false>
  static void run_logits_impl(
      const tvm::ffi::TensorView logits,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight,
      tvm::ffi::Optional<tvm::ffi::TensorView> x_q,
      tvm::ffi::Optional<tvm::ffi::TensorView> x_sf) {
    using namespace host;
    auto num_tokens = SymbolicSize{"M"};
    auto hidden_size = SymbolicSize{"H"};
    auto num_experts = SymbolicSize{"E"};
    auto device_ref = SymbolicDevice{};
    auto hidden_dtype = SymbolicDType{};

    TensorMatcher({num_tokens, num_experts})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(logits);
    TensorMatcher({num_tokens, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(hidden_dtype)
        .verify(hidden);
    {
      const auto dt = hidden_dtype.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16,
                   "MiniMax route logits hidden must be bf16");
    }
    TensorMatcher({num_experts, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(gate_weight);

    const auto rows = static_cast<int>(num_tokens.unwrap());
    const auto experts = static_cast<int>(num_experts.unwrap());
    const auto hsize = static_cast<int>(hidden_size.unwrap());
    if (rows == 0) return;
    RuntimeCheck(experts > 0 && experts <= kMaxExperts,
                 "MiniMax route logits supports 1..256 experts");
    RuntimeCheck(hsize % 128 == 0,
                 "MiniMax route external quant requires hidden_size % 128 == 0");

    uint8_t* x_q_ptr = nullptr;
    uint8_t* x_sf_ptr = nullptr;
    if (x_q.has_value()) {
      RuntimeCheck(x_sf.has_value(), "x_sf is required with x_q");
      x_q_ptr = static_cast<uint8_t*>(x_q.value().data_ptr());
      x_sf_ptr = static_cast<uint8_t*>(x_sf.value().data_ptr());
    }

    const auto stream = LaunchKernel::resolve_device(device_ref.unwrap());
    const dim3 grid_dot(rows, (experts + kWarps - 1) / kWarps);
    minimax_route_fp32_logits_tile_shared_kernel<kWarps, kApplySigmoid><<<
        grid_dot, kWarpSize * kWarps, hsize * sizeof(__nv_bfloat16), stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data_ptr()),
        static_cast<const float*>(gate_weight.data_ptr()),
        static_cast<float*>(logits.data_ptr()),
        x_q_ptr, x_sf_ptr,
        rows, hsize, experts);
  }
};
