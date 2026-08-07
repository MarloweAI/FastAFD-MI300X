#pragma once
// HIP shim for <nccl.h>. This directory is on the JIT include path only when
// torch is a ROCm build (kernel/utils.py: DEFAULT_INCLUDE / IS_HIP), so it
// cannot shadow real NCCL on an NVIDIA build.
//
// RCCL is API-compatible with NCCL for the HOST-side surface that src/pynccl.cu
// uses, including the window/symmetric-memory registration calls:
//   ncclCommInitRank, ncclAllReduce, ncclAllGather, ncclCommAbort,
//   ncclCommGetAsyncError, ncclMemAlloc/Free,
//   ncclCommWindowRegister/Deregister, ncclWindow_t
// -- all declared in rccl.h and exported from librccl.so on ROCm 7.2.4.
//
// NOT provided by RCCL: the DEVICE-side API (nccl_device.h -- ncclDevComm_t,
// ncclDevCommCreate, ncclGetLsaDevicePointer, ncclTeamLsa/ncclTeamWorld). That is
// what the vendored DeepEP kernels are written against, and why DeepEP itself
// cannot be ported this way. See dev_log/qwen/02_dependency_inventory.md sec E and
// dev_log/qwen/10_afd_transport_options.md.
#include <rccl/rccl.h>
