#pragma once
// HIP shim for <cuda_fp16.h>. See cuda_bf16.h for why this directory exists.
// ROCm provides __half / __half2 and the __float2half / __half2float
// intrinsics under the same names, so this is a pure redirect.
#include <hip/hip_fp16.h>
