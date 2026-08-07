#pragma once
// HIP shim for <cub/cub.cuh>.
//
// This directory is on the JIT include path only when torch is a ROCm build
// (kernel/utils.py: DEFAULT_INCLUDE / IS_HIP), so it cannot shadow real CUB on
// an NVIDIA build.
//
// hipCUB is AMD's CUB-compatible layer over rocPRIM and ships with ROCm at
// /opt/rocm/include/hipcub. Its API mirrors CUB's under the `hipcub` namespace,
// so aliasing the namespace is enough for the symbols minisgl uses:
// cub::BlockReduce, cub::Max, cub::ArgMax, cub::KeyValuePair
// (jit/moe_topk_softmax.cu, jit/minimax_route_fused.cu).
#include <hipcub/hipcub.hpp>

namespace cub = hipcub;
