#pragma once

// CUDA -> HIP compatibility shim for the minisgl in-house JIT kernels.
//
// Scope: only the host-side runtime/launch API that `minisgl/utils.cuh` uses.
// This is deliberately NOT a general hipify layer — device-side PTX (TMA,
// mbarrier, warpgroup, PDL) has no gfx942 equivalent and is handled at the
// call sites instead (see dev_log/qwen/02_dependency_inventory.md §C).
//
// `tvm_ffi` already selects hipcc and `--offload-arch` on its own when torch is
// a ROCm build, so all that is needed here is the type/function name mapping.

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(USE_ROCM)

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// --- error handling ---
using cudaError_t = hipError_t;
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

// --- streams ---
using cudaStream_t = hipStream_t;

// --- function attributes ---
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize \
  hipFuncAttributeMaxDynamicSharedMemorySize

// --- memcpy (used by src/pynccl.cu) ---
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy hipMemcpy
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToHost hipMemcpyHostToHost
#define cudaMemsetAsync hipMemsetAsync
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize

// --- extended launch (hipLaunchKernelEx exists in ROCm >= 6.2) ---
using cudaLaunchConfig_t = hipLaunchConfig_t;
using cudaLaunchAttribute = hipLaunchAttribute;
#define cudaLaunchKernelEx hipLaunchKernelEx

// --- DLPack device type ---
// torch on ROCm reports kDLROCM (10), not kDLCUDA (2), for GPU tensors -- verified:
// `torch.randn(8, device='cuda').__dlpack_device__()` -> (kDLROCM, 0), and tvm_ffi
// prints the device as `rocm:0`. Code doing an explicit device_type comparison must
// use this macro instead of hard-coding kDLCUDA.
//
// NOTE: `TensorMatcher::with_device<kDLCUDA>(ref)` in minisgl/tensor.h does NOT need
// changing -- its DeviceRef::verify() only compares when a value is already bound and
// otherwise adopts the tensor's device, so the template option list is never enforced.
// That check constrains tensors to agree with each other, not to be CUDA.
#define MINISGL_DL_GPU_DEVICE kDLROCM

// --- kernel parameter attributes ---
// `__grid_constant__` (CUDA 12+) marks a by-value kernel parameter as uniform
// across the grid, letting nvcc keep it in constant memory. It is a pure
// optimization hint with no semantic effect and no HIP equivalent, so it is
// safe to erase. Used by jit/{store,index,qk_norm_rope}.cu.
#define __grid_constant__

// --- Programmatic Dependent Launch ---
// PDL (`griddepcontrol` / programmaticStreamSerializationAllowed) is an SM90+
// feature with no CDNA3 equivalent: HIP defines neither the launch attribute
// nor the device intrinsics. Device-side use is already template-gated on
// `kUsePDL` in utils.cuh, so it compiles out when use_pdl=false. Host-side, we
// define the enumerator to an invalid value and static_assert at the (single)
// use site rather than silently launching without the attribute.
#define MINISGL_HIP_NO_PDL 1

#else  // not ROCm

// On CUDA the DLPack GPU device type is kDLCUDA; see the ROCm branch above.
#define MINISGL_DL_GPU_DEVICE kDLCUDA

#endif  // ROCm
