// Fused (residual-add) RMSNorm + per-128-group UE8M0 FP8 quantization.
//
// Replaces the two-kernel chain `fused_add_rmsnorm` (flashinfer) followed by
// `per_token_cast_to_fp8` for the decode attention path: the normed hidden
// states feed a DeepGEMM block-scaled FP8 GEMM, so this kernel writes the FP8
// bytes + packed UE8M0 scales directly (DeepGEMM/TMA layout) and the updated
// bf16 residual, saving one full HBM round trip of the hidden states.
//
// One block per token (row): the RMS reduction spans the whole row, so the
// per-128-group quant (which the standalone cast does block-per-group) is
// folded into the same block after the row reduction.
//
// Numerics match the reference chain: RMS over (x+residual) in fp32, then the
// normed value is ROUNDED TO BF16 before the per-group amax/quant (the
// reference writes a bf16 normed tensor that the cast then re-reads).

#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarpSize * kWarps;  // 256
constexpr int kGroup = 128;                   // K granularity

// One block per token.  H must be a multiple of 128*kWarps = 512 so every warp
// owns the same number of 128-groups and every lane owns 4 contiguous elems.
__launch_bounds__(kThreads) __global__ void add_rmsnorm_fp8_quant_kernel(
    __nv_bfloat16* __restrict__ residual,  // [m, H] in/out (x+res written back)
    const __nv_bfloat16* __restrict__ input,   // [m, H]
    const __nv_bfloat16* __restrict__ weight,  // [H]
    uint8_t* __restrict__ out_q,               // [m, H] e4m3 bytes
    uint32_t* __restrict__ out_s,              // packed UE8M0, stride (1, tma_mn)
    const int H, const int groups, const int tma_aligned_mn,
    const float eps) {
  const int row = blockIdx.x;
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int groups_per_warp = groups / kWarps;   // e.g. 32/4 = 8

  const __nv_bfloat16* x_row = input + static_cast<int64_t>(row) * H;
  __nv_bfloat16* res_row = residual + static_cast<int64_t>(row) * H;
  uint8_t* q_row = out_q + static_cast<int64_t>(row) * H;

  // Each thread owns 4 contiguous elems in each of its groups_per_warp groups.
  // group index g_k = warp + j*kWarps (j=0..groups_per_warp-1); within group the
  // 4 elems are [lane*4 .. lane*4+3].
  float s[8 /*max groups_per_warp*/][4];

  // ---- phase 1: residual add + sum of squares ----------------------------
  float ss = 0.f;
  #pragma unroll 1
  for (int j = 0; j < groups_per_warp; ++j) {
    const int g_k = warp + j * kWarps;
    const int base = g_k * kGroup + lane * 4;
    const uint2 xv = *reinterpret_cast<const uint2*>(x_row + base);
    const uint2 rv = *reinterpret_cast<const uint2*>(res_row + base);
    const __nv_bfloat16* xh = reinterpret_cast<const __nv_bfloat16*>(&xv);
    const __nv_bfloat16* rh = reinterpret_cast<const __nv_bfloat16*>(&rv);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float v = __bfloat162float(xh[i]) + __bfloat162float(rh[i]);
      s[j][i] = v;
      ss += v * v;
    }
  }

  // block reduction of ss
  __shared__ float warp_ss[kWarps];
  #pragma unroll
  for (int off = kWarpSize / 2; off > 0; off >>= 1)
    ss += __shfl_xor_sync(0xffffffffu, ss, off);
  if (lane == 0) warp_ss[warp] = ss;
  __syncthreads();
  float total = 0.f;
  #pragma unroll
  for (int w = 0; w < kWarps; ++w) total += warp_ss[w];
  const float rms = rsqrtf(total / H + eps);

  // ---- phase 2: write residual, normalize, per-group quant ---------------
  #pragma unroll 1
  for (int j = 0; j < groups_per_warp; ++j) {
    const int g_k = warp + j * kWarps;
    const int base = g_k * kGroup + lane * 4;
    const uint2 wv = *reinterpret_cast<const uint2*>(weight + g_k * kGroup + lane * 4);
    const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wv);

    // residual writeback (x+res rounded to bf16) and bf16-rounded normed value
    __nv_bfloat16 res_out[4];
    float normed[4];
    float amax = eps;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      res_out[i] = __float2bfloat16(s[j][i]);
      // round normed through bf16 exactly like the reference (flashinfer
      // writes bf16, the cast re-reads it)
      const float nv = s[j][i] * rms * __bfloat162float(wh[i]);
      const float nb = __bfloat162float(__float2bfloat16(nv));
      normed[i] = nb;
      amax = fmaxf(amax, fabsf(nb));
    }
    *reinterpret_cast<uint2*>(res_row + base) = *reinterpret_cast<uint2*>(res_out);

    // group amax across the 32 lanes of this warp
    #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));

    // UE8M0 ceil scale (identical to per_token_fp8_packed.cu)
    float y_s = amax / 448.0f;
    y_s = fmaxf(y_s, 1e-10f);
    const uint32_t bits = __float_as_uint(y_s);
    const uint8_t exp_byte = static_cast<uint8_t>(
        ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1u : 0u));
    const float inv_y = 1.0f / __uint_as_float(static_cast<uint32_t>(exp_byte) << 23);

    // quant store 4 elems
    __nv_fp8x2_storage_t out2[2];
    #pragma unroll
    for (int i = 0; i < 4; i += 2)
      out2[i / 2] = __nv_cvt_float2_to_fp8x2(
          make_float2(normed[i] * inv_y, normed[i + 1] * inv_y),
          __NV_SATFINITE, __NV_E4M3);
    *reinterpret_cast<uint32_t*>(q_row + base) =
        *reinterpret_cast<const uint32_t*>(out2);

    // packed scale byte: out_s laid out (packed_groups, tma_aligned_mn) int32,
    // byte position = g_k % 4, int32 index = (g_k/4)*tma_aligned_mn + row
    if (lane == 0) {
      const int pack_idx = g_k / 4;
      const int pos = g_k % 4;
      const int64_t out_idx =
          static_cast<int64_t>(pack_idx) * tma_aligned_mn + row;
      reinterpret_cast<uint8_t*>(out_s)[out_idx * 4 + pos] = exp_byte;
    }
  }
}

}  // namespace

struct AddRMSNormFp8QuantKernel {
  static void run(
      const tvm::ffi::TensorView residual,  // [m, H] bf16, in/out
      const tvm::ffi::TensorView input,     // [m, H] bf16
      const tvm::ffi::TensorView weight,    // [H] bf16
      const tvm::ffi::TensorView out_q,     // [m, H] uint8
      const tvm::ffi::TensorView out_s,     // [m, packed_groups] int32
      double eps) {
    using namespace host;
    auto m = SymbolicSize{"M"};
    auto Hd = SymbolicSize{"H"};
    auto device_ref = SymbolicDevice{};
    auto bf16 = SymbolicDType{};

    TensorMatcher({m, Hd}).with_device<kDLCUDA>(device_ref).with_dtype(bf16).verify(residual);
    TensorMatcher({m, Hd}).with_device<kDLCUDA>(device_ref).with_dtype(bf16).verify(input);
    TensorMatcher({Hd}).with_device<kDLCUDA>(device_ref).with_dtype(bf16).verify(weight);
    {
      const auto dt = bf16.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16,
                   "residual/input/weight must be bf16");
    }

    const int rows = static_cast<int>(m.unwrap());
    const int H = static_cast<int>(Hd.unwrap());
    if (rows == 0) return;
    RuntimeCheck(H % (kGroup * kWarps) == 0,
                 "add_rmsnorm_fp8_quant needs H % 512 == 0");
    RuntimeCheck(H <= kGroup * kWarps * 8,
                 "add_rmsnorm_fp8_quant needs H <= 4096 (reg array limit)");
    const int groups = H / kGroup;
    const int tma_aligned_mn = (rows + 3) / 4 * 4;

    const auto stream = LaunchKernel::resolve_device(device_ref.unwrap());
    add_rmsnorm_fp8_quant_kernel<<<rows, kThreads, 0, stream>>>(
        static_cast<__nv_bfloat16*>(residual.data_ptr()),
        static_cast<const __nv_bfloat16*>(input.data_ptr()),
        static_cast<const __nv_bfloat16*>(weight.data_ptr()),
        static_cast<uint8_t*>(out_q.data_ptr()),
        static_cast<uint32_t*>(out_s.data_ptr()),
        H, groups, tma_aligned_mn, static_cast<float>(eps));
  }
};
