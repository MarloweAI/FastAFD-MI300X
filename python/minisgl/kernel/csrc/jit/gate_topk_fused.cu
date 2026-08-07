// Fused router: gate GEMM (bf16, TENSOR CORES) + softmax + top-k
// (+ optional renormalize) in ONE kernel, replacing the cublas gate GEMM +
// splitK reduce + topk_gating launches on the per-layer decode path.
//
// v2: the gate GEMM runs on tensor cores via mma.sync.m16n8k16 (the v1
// CUDA-core GEMV measured 53us/layer vs cublas nvjet's ~5us — never move a
// GEMM off the tensor cores).  Mapping: one thread block per 16-token tile;
// the block owns ALL experts, so the top-k epilogue needs no cross-block
// sync: cp.async double-buffered K loop -> ldmatrix -> mma -> logits in
// shared memory -> per-token warp top-k (iterative warp arg-max).

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
inline constexpr int kTokensPerBlock = 16;  // MMA M
inline constexpr int kKChunk = 64;          // K tile per stage
inline constexpr int kRowPad = 72;         // chunk + 8 pad
inline constexpr int kMaxExperts = 128;     // one block owns all experts
inline constexpr int kTPB = 256;            // 8 warps
inline constexpr int kNTilesPerWarp = kMaxExperts / 8 / (kTPB / kWarpSize);

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)),
               "l"(src));
}

__device__ __forceinline__ void cp_async_16_zfill(void* dst, const void* src) {
  // src-size 0 => zero-fill the 16 bytes
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n" ::"r"(smem_u32(dst)),
               "l"(src));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// Release-acquire ticket replaces two full __threadfence()s (expensive on
// GB200): syncthreads collects the block's writes, the acq_rel increment
// publishes them and acquires the other blocks' writes for the reader.
__device__ __forceinline__ int atom_add_acqrel_gpu(int* p, int v) {
  int old;
  asm volatile("atom.add.acq_rel.gpu.global.s32 %0, [%1], %2;"
               : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}

inline constexpr int kNSplit = 16;  // expert-parallel blocks per token tile (8 experts each)
inline constexpr int kStages = 4;   // cp.async pipeline depth (latency-bound kernel)


__launch_bounds__(kTPB) __global__ void gate_topk_fused_tc_kernel(
    const __nv_bfloat16* __restrict__ hidden,  // [num_tokens, hidden_size]
    const __nv_bfloat16* __restrict__ gate_w,  // [num_experts, hidden_size]
    float* __restrict__ topk_weights,          // [num_tokens, topk]
    int32_t* __restrict__ topk_ids,            // [num_tokens, topk]
    float* __restrict__ logits_gmem,           // [mtiles, 16, kMaxExperts]
    int* __restrict__ counters,                // [mtiles], zero-initialized
    uint8_t* __restrict__ x_q,                 // optional [num_tokens, hidden] e4m3
    uint8_t* __restrict__ x_sf,                // optional [num_tokens, hidden/32] ue8m0
    const int num_tokens, const int num_experts, const int hidden_size,
    const int topk, const bool renormalize) {
  // v3 layout: grid = mtiles x kNSplit blocks.  A block owns 8 experts with
  // the FULL K (no cross-block reduction); its 8 warps split K and run
  // fully independent per-warp cp.async pipelines (cp.async groups are
  // per-thread state, so the main loop needs no __syncthreads at all),
  // then reduce in shared memory.  The last block of each token tile runs
  // the top-k over the assembled [16, num_experts] logits.
  constexpr int kNumWarps = kTPB / kWarpSize;          // 8
  constexpr int kWarpABytes = kStages * kTokensPerBlock * kRowPad * 2;
  constexpr int kWarpBBytes = kStages * 8 * kRowPad * 2;
  extern __shared__ char smem_raw[];
  // per-warp staging: [warp][A stages][16][kRowPad] + [warp][B stages][8][kRowPad]
  auto a_smem = [&](int w, int stage, int row, int col) -> __nv_bfloat16& {
    return reinterpret_cast<__nv_bfloat16*>(smem_raw + w * (kWarpABytes + kWarpBBytes))
        [(stage * kTokensPerBlock + row) * kRowPad + col];
  };
  auto b_smem = [&](int w, int stage, int row, int col) -> __nv_bfloat16& {
    return reinterpret_cast<__nv_bfloat16*>(smem_raw + w * (kWarpABytes + kWarpBBytes) + kWarpABytes)
        [(stage * 8 + row) * kRowPad + col];
  };
  // reduce area after the staging region
  float* red_smem = reinterpret_cast<float*>(
      smem_raw + kNumWarps * (kWarpABytes + kWarpBBytes));  // [kNumWarps][16][8]

  const int mtile = blockIdx.x / kNSplit;
  const int nsplit = blockIdx.x % kNSplit;
  const int m0 = mtile * kTokensPerBlock;
  const int n_base = nsplit * 8;  // this block's 8 experts
  const int tid = threadIdx.x;
  const int warp = tid / kWarpSize;
  const int lane = tid % kWarpSize;

  // Per-warp K slice
  const int k_slice = hidden_size / kNumWarps;
  const int k_begin = warp * k_slice;

  // Per-warp async loader: A 16x64 (8 chunks of 16B per lane covers 128
  // 16B-chunks with 32 lanes x4) + B 8x64 (64 chunks, 32 lanes x2)
  constexpr int kChunks16B = kKChunk / 8;  // 16B chunks per row per stage
  const auto load_stage = [&](int stage, int kc) {
    #pragma unroll
    for (int i = 0; i < kTokensPerBlock * kChunks16B / kWarpSize; ++i) {
      const int idx = lane + i * kWarpSize;
      const int row = idx / kChunks16B, chunk = idx % kChunks16B;
      void* dst = &a_smem(warp, stage, row, chunk * 8);
      if (m0 + row < num_tokens)
        cp_async_16(dst, hidden + static_cast<size_t>(m0 + row) * hidden_size + kc + chunk * 8);
      else
        cp_async_16_zfill(dst, hidden);
    }
    #pragma unroll
    for (int i = 0; i < 8 * kChunks16B / kWarpSize; ++i) {
      const int idx = lane + i * kWarpSize;
      const int row = idx / kChunks16B, chunk = idx % kChunks16B;
      cp_async_16(&b_smem(warp, stage, row, chunk * 8),
                  gate_w + static_cast<size_t>(n_base + row) * hidden_size + kc + chunk * 8);
    }
    cp_async_commit();
  };

  float acc[4] = {};
  const int num_chunks = k_slice / kKChunk;

  // Optional fused per-32 FP8 quant of the hidden states (the gate GEMV
  // streams every token row through smem anyway).  Block `nsplit` owns the
  // k-slice [nsplit*hidden/kNSplit, ...), which lives entirely inside warp
  // nsplit/2's pipeline; lane <-> (token, 32-group) inside each 64-wide
  // chunk.  Bit-identical to the AG mega kernel's phase-0 quant.
  const int q_size = hidden_size / kNSplit;                  // bytes of K per block
  const bool q_on = (x_q != nullptr) && (warp == nsplit / 2);
  const int q_chunk_begin = (nsplit % 2) * (q_size / kKChunk);
  const int q_chunk_end = q_chunk_begin + q_size / kKChunk;
  const auto quant_chunk = [&](int stage, int c) {
    const int row = lane / 2, grp = lane % 2;
    if (m0 + row >= num_tokens) return;
    const int k0 = k_begin + c * kKChunk + grp * 32;
    float vals[32];
    #pragma unroll
    for (int j = 0; j < 32; ++j)
      vals[j] = __bfloat162float(a_smem(warp, stage, row, grp * 32 + j));
    float amax = 1e-4f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(vals[j]));
    const uint32_t bits = __float_as_uint(amax * (1.0f / 448.0f));
    const uint32_t e = ((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) ? 1u : 0u);
    x_sf[static_cast<size_t>(m0 + row) * (hidden_size / 32) + k0 / 32] =
        static_cast<uint8_t>(e);
    const float inv_scale = __uint_as_float((254u - e) << 23);
    uint32_t packed[8];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
      reinterpret_cast<__nv_fp8x2_storage_t*>(packed)[j] = __nv_cvt_float2_to_fp8x2(
          make_float2(vals[2 * j] * inv_scale, vals[2 * j + 1] * inv_scale),
          __NV_SATFINITE, __NV_E4M3);
    }
    auto dst = reinterpret_cast<uint4*>(x_q + static_cast<size_t>(m0 + row) * hidden_size + k0);
    dst[0] = *reinterpret_cast<uint4*>(packed);
    dst[1] = *reinterpret_cast<uint4*>(packed + 4);
  };
  for (int st = 0; st < kStages - 1 && st < num_chunks; ++st)
    load_stage(st, k_begin + st * kKChunk);
  for (int c = 0; c < num_chunks; ++c) {
    const int cur = c % kStages;
    if (c + kStages - 1 < num_chunks)
      load_stage((c + kStages - 1) % kStages, k_begin + (c + kStages - 1) * kKChunk);
    else
      cp_async_commit();
    cp_async_wait<kStages - 1>();
    __syncwarp();

    #pragma unroll
    for (int ks = 0; ks < kKChunk / 16; ++ks) {
      uint32_t a0, a1, a2, a3;
      {
        const int grp = lane / 8;
        const int row = (grp % 2) * 8 + (lane % 8);
        const int col = ks * 16 + (grp / 2) * 8;
        const uint32_t addr = smem_u32(&a_smem(warp, cur, row, col));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "r"(addr));
      }
      uint32_t b0, b1;
      {
        const int r = lane % 8;
        const int half = (lane / 8) % 2;
        const uint32_t addr = smem_u32(&b_smem(warp, cur, r, ks * 16 + half * 8));
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(b0), "=r"(b1)
            : "r"(addr));
      }
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    if (q_on && c >= q_chunk_begin && c < q_chunk_end)
      quant_chunk(cur, c);
    __syncwarp();
  }

  // ---- intra-block K reduction in shared memory --------------------------
  {
    const int row = lane / 4;
    const int col = (lane % 4) * 2;
    float* mine = red_smem + warp * kTokensPerBlock * 8;
    mine[row * 8 + col] = acc[0];
    mine[row * 8 + col + 1] = acc[1];
    mine[(row + 8) * 8 + col] = acc[2];
    mine[(row + 8) * 8 + col + 1] = acc[3];
  }
  __syncthreads();
  // 16x8 = 128 outputs; thread t sums over the 8 warps
  float* block_logits = logits_gmem +
      (static_cast<size_t>(mtile) * kTokensPerBlock) * kMaxExperts;
  if (tid < kTokensPerBlock * 8) {
    float v = 0.f;
    #pragma unroll
    for (int w = 0; w < kNumWarps; ++w)
      v += red_smem[w * kTokensPerBlock * 8 + tid];
    block_logits[(tid / 8) * kMaxExperts + n_base + (tid % 8)] = v;
  }

  return;
}

// Second kernel: top-k over the assembled logits (one block per token tile,
// one warp per token).  The kernel boundary is the cheapest device-wide
// barrier — much faster than an in-kernel ticket + memory fence round.
__launch_bounds__(kTokensPerBlock * kWarpSize) __global__ void gate_topk_select_kernel(
    const float* __restrict__ logits_gmem,  // [mtiles, 16, kMaxExperts]
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    float* __restrict__ w_stage,            // optional symm-buffer staging copy
    const int num_tokens, const int num_experts, const int topk,
    const bool renormalize) {
  const int mtile = blockIdx.x;
  const int m0 = mtile * kTokensPerBlock;
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const float* block_logits =
      logits_gmem + static_cast<size_t>(mtile) * kTokensPerBlock * kMaxExperts;

  constexpr int kMaxExpertsPerLane = kMaxExperts / kWarpSize;  // 4
  const int per_lane = (num_experts + kWarpSize - 1) / kWarpSize;
  {
    const int t = warp;
    const int token = m0 + t;
    if (token >= num_tokens) return;

    float logit[kMaxExpertsPerLane];
    #pragma unroll
    for (int i = 0; i < kMaxExpertsPerLane; ++i) {
      const int e = i * kWarpSize + lane;
      logit[i] = (i < per_lane && e < num_experts)
                     ? block_logits[t * kMaxExperts + e]
                     : -FLT_MAX;
    }

    float m_all = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < kMaxExpertsPerLane; ++i) m_all = fmaxf(m_all, logit[i]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      m_all = fmaxf(m_all, __shfl_xor_sync(0xffffffffu, m_all, off));
    float denom_all = 0.f;
    #pragma unroll
    for (int i = 0; i < kMaxExpertsPerLane; ++i)
      denom_all += logit[i] > -FLT_MAX ? __expf(logit[i] - m_all) : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      denom_all += __shfl_xor_sync(0xffffffffu, denom_all, off);

    uint32_t used = 0;
    float my_logit = -FLT_MAX;
    int my_expert = -1;
    float top0 = -FLT_MAX;
    for (int j = 0; j < topk; ++j) {
      float best = -FLT_MAX;
      int best_i = -1;
      #pragma unroll
      for (int i = 0; i < kMaxExpertsPerLane; ++i) {
        if (!((used >> i) & 1u) && logit[i] > best) {
          best = logit[i];
          best_i = i;
        }
      }
      float v = best;
      uint32_t src = lane;
      #pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, off);
        const uint32_t os = __shfl_xor_sync(0xffffffffu, src, off);
        if (ov > v || (ov == v && os < src)) {
          v = ov;
          src = os;
        }
      }
      const int wbi = __shfl_sync(0xffffffffu, best_i, src);
      if (lane == static_cast<int>(src)) used |= 1u << wbi;
      if (j == 0) top0 = v;
      if (lane == j) {
        my_logit = v;
        my_expert = wbi * kWarpSize + static_cast<int>(src);
      }
    }

    float w = lane < topk ? __expf(my_logit - (renormalize ? top0 : m_all)) : 0.f;
    float w_sum = w;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      w_sum += __shfl_xor_sync(0xffffffffu, w_sum, off);
    if (lane < topk) {
      topk_ids[static_cast<size_t>(token) * topk + lane] = my_expert;
      const float w_out = renormalize ? w / w_sum : w / denom_all;
      topk_weights[static_cast<size_t>(token) * topk + lane] = w_out;
      if (w_stage != nullptr)
        w_stage[static_cast<size_t>(token) * topk + lane] = w_out;
    }
  }
}

}  // namespace

struct GateTopkFusedKernel {
  static void run(
      const tvm::ffi::TensorView topk_weights,
      const tvm::ffi::TensorView topk_indices,
      const tvm::ffi::TensorView hidden,
      const tvm::ffi::TensorView gate_weight,
      const tvm::ffi::TensorView partials,
      const tvm::ffi::TensorView counters,
      bool renormalize,
      tvm::ffi::Optional<tvm::ffi::TensorView> x_q,
      tvm::ffi::Optional<tvm::ffi::TensorView> x_sf,
      tvm::ffi::Optional<tvm::ffi::TensorView> w_stage) {
    using namespace host;
    auto num_tokens = SymbolicSize{"M"};
    auto topk = SymbolicSize{"K"};
    auto num_experts = SymbolicSize{"E"};
    auto hidden_size = SymbolicSize{"H"};
    auto device_ref = SymbolicDevice{};
    auto data_dtype_ = SymbolicDType{};

    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<float>()
        .verify(topk_weights);
    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t>()
        .verify(topk_indices);
    TensorMatcher({num_tokens, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(data_dtype_)
        .verify(hidden);
    TensorMatcher({num_experts, hidden_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype(data_dtype_)
        .verify(gate_weight);
    {
      const auto dt = data_dtype_.unwrap();
      RuntimeCheck(dt.code == DLDataTypeCode::kDLBfloat && dt.bits == 16,
                   "hidden/gate_weight must be bf16");
    }

    const auto rows = static_cast<int>(num_tokens.unwrap());
    const auto experts = static_cast<int>(num_experts.unwrap());
    const auto hsize = static_cast<int>(hidden_size.unwrap());
    const auto topk_value = static_cast<int>(topk.unwrap());
    if (rows == 0) return;
    RuntimeCheck(hsize % (kKChunk * (kTPB / kWarpSize)) == 0, "hidden_size must be divisible by 512");
    RuntimeCheck(experts % 8 == 0 && experts <= kMaxExperts,
                 "num_experts must be a multiple of 8 and <= 128");
    RuntimeCheck(topk_value <= 32, "topk must be <= 32");

    const auto stream = LaunchKernel::resolve_device(device_ref.unwrap());
    const int num_mtiles = (rows + kTokensPerBlock - 1) / kTokensPerBlock;
    constexpr int kNumWarps = kTPB / kWarpSize;
    constexpr int kSmemBytes =
        kNumWarps * kStages * (kTokensPerBlock + 8) * kRowPad * 2 +
        kNumWarps * kTokensPerBlock * 8 * static_cast<int>(sizeof(float));
    static bool attr_set = false;
    if (!attr_set) {
      cudaFuncSetAttribute(gate_topk_fused_tc_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
      attr_set = true;
    }
    uint8_t* x_q_ptr = nullptr;
    uint8_t* x_sf_ptr = nullptr;
    float* w_stage_ptr = nullptr;
    if (x_q.has_value()) {
      RuntimeCheck(x_sf.has_value(), "x_sf required with x_q");
      RuntimeCheck(hsize % (kNSplit * kKChunk) == 0,
                   "fused quant needs hidden_size % 1024 == 0");
      x_q_ptr = static_cast<uint8_t*>(x_q.value().data_ptr());
      x_sf_ptr = static_cast<uint8_t*>(x_sf.value().data_ptr());
    }
    if (w_stage.has_value())
      w_stage_ptr = static_cast<float*>(w_stage.value().data_ptr());
    gate_topk_fused_tc_kernel<<<num_mtiles * kNSplit, kTPB, kSmemBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(hidden.data_ptr()),
        static_cast<const __nv_bfloat16*>(gate_w_ptr(gate_weight)),
        static_cast<float*>(topk_weights.data_ptr()),
        static_cast<int32_t*>(topk_indices.data_ptr()),
        static_cast<float*>(partials.data_ptr()),
        static_cast<int*>(counters.data_ptr()),
        x_q_ptr, x_sf_ptr,
        rows, experts, hsize, topk_value, renormalize);
    gate_topk_select_kernel<<<num_mtiles, kTokensPerBlock * kWarpSize, 0, stream>>>(
        static_cast<const float*>(partials.data_ptr()),
        static_cast<float*>(topk_weights.data_ptr()),
        static_cast<int32_t*>(topk_indices.data_ptr()),
        w_stage_ptr,
        rows, experts, topk_value, renormalize);
  }

 private:
  static const void* gate_w_ptr(const tvm::ffi::TensorView& t) {
    return t.data_ptr();
  }
};
