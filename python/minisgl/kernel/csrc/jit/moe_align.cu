#include <minisgl/tensor.h>
#include <minisgl/utils.cuh>
#include <minisgl/utils.h>

#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace {

inline constexpr auto next_pow2(std::size_t value) -> std::size_t {
  std::size_t result = 1;
  while (result < value) {
    result <<= 1;
  }
  return result;
}

template <typename T>
__global__ void count_and_sort_expert_tokens_kernel(
    const T* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer, std::size_t numel) {
  const auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

  for (auto i = tid; i < numel; i += stride) {
    const auto expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    const auto rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
  }
}

__device__ __forceinline__ auto warp_exclusive_scan(
    int value, device::warp_mask_t mask = device::kFullWarpMask) -> int {
  const auto original = value;
#pragma unroll
  for (int offset = 1; offset < static_cast<int>(device::kWaveThreads); offset <<= 1) {
    const auto other = __shfl_up_sync(mask, value, offset);
    if ((threadIdx.x & (device::kWaveThreads - 1)) >= offset) {
      value += other;
    }
  }
  return value - original;
}

template <typename T>
__global__ void moe_align_block_size_kernel(
    const T* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, std::size_t numel, int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids, int32_t scan_size,
    int32_t max_num_tokens_padded) {
  if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      for (int32_t i = threadIdx.x; i < max_num_tokens_padded; i += blockDim.x) {
        sorted_token_ids[i] = static_cast<int32_t>(numel);
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  auto* shared_counts = smem;
  auto* prefix = shared_counts + num_experts;
  auto* scan_buf = prefix + num_experts + 1;
  __shared__ int32_t s_total_tokens_post_pad;

  const auto tid = static_cast<std::size_t>(threadIdx.x);
  const auto stride = static_cast<std::size_t>(blockDim.x);

  if (tid < static_cast<std::size_t>(num_experts)) {
    shared_counts[tid] = 0;
  }
  __syncthreads();

  for (auto i = tid; i < numel; i += stride) {
    const auto expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    atomicAdd(&shared_counts[expert_id], 1);
  }
  __syncthreads();

  int32_t padded_count = 0;
  if (tid < static_cast<std::size_t>(num_experts)) {
    const auto count = shared_counts[tid];
    padded_count = device::div_ceil(count, block_size) * block_size;
    scan_buf[tid] = padded_count;
  }

  auto* warp_sums = scan_buf + scan_size;
  const auto warp_id = static_cast<int>(tid / device::kWaveThreads);
  const auto lane_id = static_cast<int>(tid & (device::kWaveThreads - 1));
  const auto num_warps_for_scan =
      (scan_size + static_cast<int>(device::kWaveThreads) - 1) /
      static_cast<int>(device::kWaveThreads);

  const auto warp_sum = warp_exclusive_scan(padded_count) + padded_count;
  if (lane_id == static_cast<int>(device::kWaveThreads) - 1) {
    warp_sums[warp_id] = warp_sum;
  }
  __syncthreads();

  if (tid < device::kWaveThreads) {
    const auto value =
        (tid < static_cast<std::size_t>(num_warps_for_scan)) ? warp_sums[tid] : 0;
    const auto inclusive = warp_exclusive_scan(value) + value;
    warp_sums[tid] = inclusive;
  }
  __syncthreads();

  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  if (tid >= static_cast<std::size_t>(num_experts) &&
      tid < static_cast<std::size_t>(scan_size)) {
    scan_buf[tid] = 0;
  }
  __syncthreads();

  const auto value =
      (tid < static_cast<std::size_t>(scan_size)) ? scan_buf[tid] : 0;
  const auto prefix_sum = warp_exclusive_scan(value);
  if (lane_id == static_cast<int>(device::kWaveThreads) - 1) {
    warp_sums[warp_id] = prefix_sum + value;
  }
  __syncthreads();

  if (warp_id == 0) {
    const auto warp_value =
        (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = warp_exclusive_scan(warp_value);
  }
  __syncthreads();

  const auto offset = warp_sums[warp_id];
  if (tid < static_cast<std::size_t>(scan_size)) {
    scan_buf[tid] = prefix_sum + offset;
  }
  __syncthreads();

  if (tid < static_cast<std::size_t>(num_experts)) {
    prefix[tid] = scan_buf[tid];
  }
  __syncthreads();

  if (tid <= static_cast<std::size_t>(num_experts)) {
    cumsum[tid] = prefix[tid];
  }

  const auto num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = static_cast<int32_t>(tid); i < num_blocks; i += blockDim.x) {
    const auto block_start = i * block_size;
    int32_t left = 0;
    int32_t right = num_experts;
    while (left < right) {
      const auto mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 2;
  }
}

template <typename T, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const T* __restrict__ topk_ids, int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad, int32_t num_experts,
    int32_t block_size, std::size_t numel, bool pad_sorted_token_ids,
    int32_t max_num_tokens_padded) {
  if (threadIdx.x < fill_threads) {
    if (pad_sorted_token_ids) {
      for (int32_t i = threadIdx.x; i < max_num_tokens_padded; i += fill_threads) {
        sorted_token_ids[i] = static_cast<int32_t>(numel);
      }
    }
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const auto tid = static_cast<std::size_t>(threadIdx.x - fill_threads);
  const auto stride = static_cast<std::size_t>(blockDim.x - fill_threads);

  extern __shared__ int32_t shared_mem[];
  auto* cumsum = shared_mem;
  auto* tokens_cnts = shared_mem + num_experts + 1;

  for (int32_t i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (auto i = tid; i < numel; i += stride) {
    const auto expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    ++tokens_cnts[(tid + 1) * num_experts + expert_id];
  }
  __syncthreads();

  if (tid < static_cast<std::size_t>(num_experts)) {
    tokens_cnts[tid] = 0;
    for (std::size_t i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] += tokens_cnts[(i - 1) * num_experts + tid];
    }
  }
  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int32_t i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  device::div_ceil(tokens_cnts[stride * num_experts + i - 1], block_size) *
                      block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }
  __syncthreads();

  if (tid < static_cast<std::size_t>(num_experts)) {
    for (int32_t i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = static_cast<int32_t>(tid) - 1;
    }
  }

  for (auto i = tid; i < numel; i += stride) {
    const auto expert_id = static_cast<int32_t>(topk_ids[i]) + 1;
    const auto rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = static_cast<int32_t>(i);
    ++tokens_cnts[tid * num_experts + expert_id];
  }
}

struct MoeAlignKernel {
  static void run(
      const tvm::ffi::TensorView topk_ids, int64_t num_experts,
      int64_t block_size, const tvm::ffi::TensorView sorted_token_ids,
      const tvm::ffi::TensorView expert_ids,
      const tvm::ffi::TensorView num_tokens_post_pad,
      const tvm::ffi::TensorView cumsum_buffer, bool pad_sorted_token_ids) {
    using namespace host;

    RuntimeCheck(num_experts > 0, "num_experts must be positive");
    RuntimeCheck(block_size > 0, "block_size must be positive");

    auto num_tokens = SymbolicSize{"M"};
    auto topk = SymbolicSize{"K"};
    auto max_num_tokens_padded = SymbolicSize{"P"};
    auto cumsum_size = SymbolicSize{"C"};
    auto topk_ids_dtype = SymbolicDType{};
    auto device_ref = SymbolicDevice{};

    TensorMatcher({num_tokens, topk})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t, int64_t>(topk_ids_dtype)
        .verify(topk_ids);
    TensorMatcher({max_num_tokens_padded})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t>()
        .verify(sorted_token_ids);
    TensorMatcher({-1})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t>()
        .verify(expert_ids);
    TensorMatcher({1})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t>()
        .verify(num_tokens_post_pad);
    TensorMatcher({cumsum_size})
        .with_device<kDLCUDA>(device_ref)
        .with_dtype<int32_t>()
        .verify(cumsum_buffer);

    RuntimeCheck(
        cumsum_size.unwrap() == num_experts + 1,
        "cumsum_buffer size mismatch: expected ", num_experts + 1, " but got ",
        cumsum_size.unwrap());

    const auto numel = static_cast<std::size_t>(num_tokens.unwrap() * topk.unwrap());
    RuntimeCheck(numel > 0, "topk_ids must be non-empty");

    const auto max_num_m_blocks =
        host::div_ceil(max_num_tokens_padded.unwrap(), block_size);
    RuntimeCheck(
        expert_ids.size(0) >= max_num_m_blocks,
        "expert_ids buffer is too small: expected at least ", max_num_m_blocks,
        " but got ", expert_ids.size(0));

    const auto cuda_device = device_ref.unwrap();
    const auto use_int32 = topk_ids_dtype.unwrap().bits == 32;
    const auto small_batch_expert_mode = (numel < 1024) && (num_experts <= 64);
    if (small_batch_expert_mode) {
      constexpr int32_t fill_threads = 256;
      const auto threads =
          std::max<int32_t>(static_cast<int32_t>(num_experts), device::kWaveThreads);
      const auto shared_mem_size = static_cast<std::size_t>(
          ((threads + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t));
      if (use_int32) {
        LaunchKernel(dim3(1), dim3(fill_threads + threads), cuda_device, shared_mem_size)(
            moe_align_block_size_small_batch_expert_kernel<int32_t, fill_threads>,
            static_cast<const int32_t*>(topk_ids.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<int32_t*>(expert_ids.data_ptr()),
            static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
            static_cast<int32_t>(num_experts), static_cast<int32_t>(block_size),
            numel, pad_sorted_token_ids,
            static_cast<int32_t>(max_num_tokens_padded.unwrap()));
      } else {
        LaunchKernel(dim3(1), dim3(fill_threads + threads), cuda_device, shared_mem_size)(
            moe_align_block_size_small_batch_expert_kernel<int64_t, fill_threads>,
            static_cast<const int64_t*>(topk_ids.data_ptr()),
            static_cast<int32_t*>(sorted_token_ids.data_ptr()),
            static_cast<int32_t*>(expert_ids.data_ptr()),
            static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
            static_cast<int32_t>(num_experts), static_cast<int32_t>(block_size),
            numel, pad_sorted_token_ids,
            static_cast<int32_t>(max_num_tokens_padded.unwrap()));
      }
      return;
    }

    const auto scan_size = next_pow2(static_cast<std::size_t>(num_experts));
    RuntimeCheck(
        scan_size <= 1024,
        "num_experts is too large for the current JIT moe_align kernel: ",
        num_experts);

    const auto threads = 1024;
    const auto shared_mem_size = static_cast<std::size_t>(
        (num_experts + (num_experts + 1) + scan_size + device::kWaveThreads) *
        sizeof(int32_t));
    const auto block_threads = std::min(256, threads);
    const auto num_blocks =
        static_cast<int>((numel + block_threads - 1) / block_threads);
    const auto actual_blocks = std::min(num_blocks, 65535);

    if (use_int32) {
      LaunchKernel(dim3(2), dim3(threads), cuda_device, shared_mem_size)(
          moe_align_block_size_kernel<int32_t>,
          static_cast<const int32_t*>(topk_ids.data_ptr()),
          static_cast<int32_t*>(sorted_token_ids.data_ptr()),
          static_cast<int32_t*>(expert_ids.data_ptr()),
          static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
          static_cast<int32_t>(num_experts), static_cast<int32_t>(block_size),
          numel, static_cast<int32_t*>(cumsum_buffer.data_ptr()),
          pad_sorted_token_ids, static_cast<int32_t>(scan_size),
          static_cast<int32_t>(max_num_tokens_padded.unwrap()));
      LaunchKernel(dim3(actual_blocks), dim3(block_threads), cuda_device)(
          count_and_sort_expert_tokens_kernel<int32_t>,
          static_cast<const int32_t*>(topk_ids.data_ptr()),
          static_cast<int32_t*>(sorted_token_ids.data_ptr()),
          static_cast<int32_t*>(cumsum_buffer.data_ptr()), numel);
    } else {
      LaunchKernel(dim3(2), dim3(threads), cuda_device, shared_mem_size)(
          moe_align_block_size_kernel<int64_t>,
          static_cast<const int64_t*>(topk_ids.data_ptr()),
          static_cast<int32_t*>(sorted_token_ids.data_ptr()),
          static_cast<int32_t*>(expert_ids.data_ptr()),
          static_cast<int32_t*>(num_tokens_post_pad.data_ptr()),
          static_cast<int32_t>(num_experts), static_cast<int32_t>(block_size),
          numel, static_cast<int32_t*>(cumsum_buffer.data_ptr()),
          pad_sorted_token_ids, static_cast<int32_t>(scan_size),
          static_cast<int32_t>(max_num_tokens_padded.unwrap()));
      LaunchKernel(dim3(actual_blocks), dim3(block_threads), cuda_device)(
          count_and_sort_expert_tokens_kernel<int64_t>,
          static_cast<const int64_t*>(topk_ids.data_ptr()),
          static_cast<int32_t*>(sorted_token_ids.data_ptr()),
          static_cast<int32_t*>(cumsum_buffer.data_ptr()), numel);
    }
  }
};

}  // namespace
