#pragma once
// HIP shim for <cuda_runtime.h>. See cuda_bf16.h for why this directory exists.
// minisgl/hip_compat.h supplies the cudaXxx -> hipXxx name mapping that the
// in-house kernels rely on; this header just pulls in the HIP runtime.
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <minisgl/hip_compat.h>
