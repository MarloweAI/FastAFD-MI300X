#pragma once

#include <minisgl/utils.h>
#include <minisgl/hip_compat.h>

#include <dlpack/dlpack.h>
#include <tvm/ffi/extra/c_env_api.h>

#include <concepts>
#include <cstddef>
#include <source_location>
#include <type_traits>

namespace device {

// Cooperative TILE width, not a hardware warp. `warp::copy`/`warp::reset` and
// their callers (jit/store.cu, jit/index.cu) only issue plain vector loads and
// stores — no cross-lane intrinsics — so this is just "how many threads share a
// row". It stays 32 on every platform: it feeds `resolve_unit_size()`, and
// widening it to 64 would raise the alignment requirement enough to trip the
// `kBytes % kBytesPerLoop` static_assert for small per-token KV sizes
// (e.g. 1 kv head x 64 dim x bf16 = 128 B), for no benefit.
inline constexpr auto kWarpThreads = 32u;

// Hardware wave/warp width. MUST be used (instead of kWarpThreads) by anything
// calling a cross-lane intrinsic — __shfl_*, __ballot_*, __reduce_* — because
// those operate on the real wavefront. CDNA/RDNA compute wave64 for gfx9xx.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(USE_ROCM)
inline constexpr auto kWaveThreads = 64u;
#else
inline constexpr auto kWaveThreads = 32u;
#endif

// Full-participation mask for the *_sync cross-lane intrinsics. HIP static_asserts
// that the mask is 64-bit (amd_warp_sync_functions.h: "The mask must be a 64-bit
// integer. Implicitly promoting a smaller integer is almost always an error."),
// so a literal 0xffffffff does not compile there — the width must follow the
// platform, hence the typedef.
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(USE_ROCM)
using warp_mask_t = unsigned long long;
inline constexpr warp_mask_t kFullWarpMask = ~warp_mask_t{0};
#else
using warp_mask_t = unsigned;
inline constexpr warp_mask_t kFullWarpMask = 0xffffffffu;
#endif

template <std::integral T, std::integral U>
__always_inline __device__ constexpr auto div_ceil(T a, U b) {
  return (a + b - 1) / b;
}

namespace pointer {

// we only allow void * pointer arithmetic for safety

template <typename T, std::integral... U>
__always_inline __device__ auto offset(T *ptr, U... offset) -> void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<char *>(ptr) + (... + offset);
}

template <typename T, std::integral... U>
__always_inline __device__ auto offset(const T *ptr, U... offset) -> const
    void * {
  static_assert(std::is_same_v<T, void>,
                "Pointer arithmetic is only allowed for void* pointers");
  return static_cast<const char *>(ptr) + (... + offset);
}

} // namespace pointer

namespace PDL {

template <bool kUsePDL> __always_inline __device__ void wait() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
}

template <bool kUsePDL> __always_inline __device__ void launch() {
  if constexpr (kUsePDL) {
    asm volatile("griddepcontrol.launch_dependents;" :::);
  }
}

} // namespace PDL

} // namespace device

namespace host {

inline auto
CUDA_CHECK(::cudaError_t error,
           std::source_location location = std::source_location::current())
    -> void {
  if (error != ::cudaSuccess) {
    [[unlikely]];
    ::host::panic(location, "CUDA error: ", ::cudaGetErrorString(error));
  }
}

inline auto
CUDA_CHECK(std::source_location location = std::source_location::current())
    -> void {
  return CUDA_CHECK(::cudaGetLastError(), location);
}

template <auto F> inline void set_smem_once(std::size_t smem_size) {
  static const auto last_smem_size = [&] {
    CUDA_CHECK(::cudaFuncSetAttribute(
        F, ::cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    return smem_size;
  }();
  RuntimeCheck(
      smem_size <= last_smem_size,
      "Dynamic shared memory size exceeds the previously set maximum size: ",
      last_smem_size, " bytes");
}

struct LaunchKernel {
public:
  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, DLDevice device,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_config(s_make_config(grid_dim, block_dim, resolve_device(device),
                               dynamic_shared_mem_bytes)) {}

  explicit LaunchKernel(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                        std::size_t dynamic_shared_mem_bytes = 0) noexcept
      : m_config(s_make_config(grid_dim, block_dim, stream,
                               dynamic_shared_mem_bytes)) {}

  static auto resolve_device(DLDevice device) -> cudaStream_t {
    return static_cast<cudaStream_t>(
        ::TVMFFIEnvGetStream(device.device_type, device.device_id));
  }

  LaunchKernel(const LaunchKernel &) = delete;
  LaunchKernel &operator=(const LaunchKernel &) = delete;

  template <typename T, typename... Args>
  auto operator()(T &&kernel, Args &&...args) const -> void {
    CUDA_CHECK(
        ::cudaLaunchKernelEx(&m_config, kernel, std::forward<Args>(args)...));
  }

  auto with_attr(bool use_pdl) -> LaunchKernel & {
    if (use_pdl) {
#ifdef MINISGL_HIP_NO_PDL
      // CDNA3 has no Programmatic Dependent Launch. Callers must pass
      // use_pdl=false on ROCm (KernelConfig.use_pdl); reaching here means a
      // kernel config was not updated for the port.
      ::host::panic(std::source_location::current(),
                    "PDL (use_pdl=true) is unsupported on ROCm/gfx942; "
                    "set KernelConfig.use_pdl=false");
#else
      m_attr_cache.id = ::cudaLaunchAttributeProgrammaticStreamSerialization;
      m_attr_cache.val.programmaticStreamSerializationAllowed = 1;
      m_config.attrs = &m_attr_cache;
      m_config.numAttrs = 1;
#endif
    } else {
      m_config.numAttrs = 0;
    }
    return *this;
  }

private:
  static auto s_make_config(dim3 grid_dim, dim3 block_dim, cudaStream_t stream,
                            std::size_t smem) -> cudaLaunchConfig_t {
    auto config = ::cudaLaunchConfig_t{};
    config.gridDim = grid_dim;
    config.blockDim = block_dim;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.numAttrs = 0;
    return config;
  }
  cudaLaunchConfig_t m_config;
  cudaLaunchAttribute m_attr_cache;
};

} // namespace host
