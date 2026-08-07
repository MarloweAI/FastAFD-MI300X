# Vendored third-party code

This directory contains vendored copies of external CUDA/C++ libraries. Each
retains its upstream license (see the `LICENSE` file in the corresponding
subdirectory). FastAFD's own modifications are noted below.

## `deepep/`

- **Upstream:** DeepEP — https://github.com/deepseek-ai/DeepEP
- **License:** MIT (Copyright (c) 2025 DeepSeek) — see `deepep/LICENSE`
- **Provenance:** a trimmed, de-branded snapshot of upstream DeepEP (the
  elastic-buffer / hybrid dispatch-combine "V2" path). The pybind module is
  renamed `_minisgl_deepep_moe`.
- **FastAFD modifications:** small local deltas only (a few lines in
  `csrc/elastic/buffer.hpp`, `deep_ep/include/deep_ep/impls/hybrid_combine.cuh`,
  and `csrc/python_api.cpp`). The CUDA kernels are otherwise upstream.

## `deepgemm/`

- **Upstream:** DeepGEMM — https://github.com/deepseek-ai/DeepGEMM
- **License:** MIT (Copyright (c) 2025 DeepSeek) — see `deepgemm/LICENSE`
- **Provenance:** a de-branded snapshot of upstream DeepGEMM. The pybind module
  is renamed `_minisgl_deepgemm`.
- **FastAFD modifications:** adds the M2N MoE megakernel family, which has no
  upstream equivalent —
  `csrc/apis/mega_m2n.hpp`,
  `csrc/jit_kernels/impls/sm100_mega_moe_m2n.hpp`,
  `deep_gemm/include/deep_gemm/impls/sm100_mega_moe_m2n.cuh`,
  `deep_gemm/include/deep_gemm/layout/mega_moe_m2n.cuh` — and modifies the
  related mega-MoE files (`csrc/apis/mega.hpp`, the `mega_moe` heuristics /
  layout / scheduler, and `deep_gemm/include/deep_gemm/comm/barrier.cuh`).

### `deepgemm/third-party/cutlass/`

- **Upstream:** NVIDIA CUTLASS — https://github.com/NVIDIA/cutlass
- **License:** BSD-3-Clause (Copyright (c) 2017-2026 NVIDIA CORPORATION &
  AFFILIATES) — see `deepgemm/third-party/cutlass/LICENSE.txt`
- Header-only dependency of DeepGEMM; unmodified.

### `deepgemm/third-party/fmt/`

- **Upstream:** {fmt} — https://github.com/fmtlib/fmt
- **License:** MIT (Copyright (c) 2012-present, Victor Zverovich and {fmt}
  contributors) — see `deepgemm/third-party/fmt/LICENSE`
- Header-only dependency of DeepGEMM; unmodified.
