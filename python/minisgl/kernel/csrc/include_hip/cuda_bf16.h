#pragma once
// HIP shim for <cuda_bf16.h>.
//
// This directory (csrc/include_hip) is added to the JIT include path ONLY when
// torch is a ROCm build (see kernel/utils.py: DEFAULT_INCLUDE / IS_HIP), so it
// cannot shadow the real CUDA headers on an NVIDIA build.
//
// ROCm's amd_hip_bf16.h already provides the conversion intrinsics with the
// same names CUDA uses (__float2bfloat16, __bfloat162float, ...) but names the
// types __hip_bfloat16 / __hip_bfloat162 rather than __nv_bfloat16 /
// __nv_bfloat162. Only the type aliases need adding.

#include <hip/hip_bf16.h>

using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;
using nv_bfloat16 = __hip_bfloat16;
using nv_bfloat162 = __hip_bfloat162;
