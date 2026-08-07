#pragma once
// HIP shim for <cuda_fp8.h>. See cuda_bf16.h for why this directory exists.
//
// INCOMPLETE ON PURPOSE. gfx942's native FP8 is the *fnuz* encoding, while the
// CUDA names below denote OCP e4m3/e5m2 (see dev_log/qwen/02_dependency_inventory.md
// §I and the T-02 result: hipBLASLt rejects float8_e4m3fn on gfx942 but accepts
// float8_e4m3fnuz). Aliasing __nv_fp8_e4m3 to the fnuz type would silently
// change numerics by a factor of 2 in the exponent bias, so it is NOT done here.
//
// The BF16 bring-up path (dev_log/qwen/03_port_plan.md M1/M2) does not use FP8 at
// all. Anything that needs these types belongs to the optional M3.5 FP8 phase
// and must handle the fn<->fnuz conversion explicitly.
#include <hip/hip_fp8.h>

#define MINISGL_HIP_FP8_SHIM_INCOMPLETE 1
