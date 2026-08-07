from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias, Union

if TYPE_CHECKING:
    from tvm_ffi import Module

def _is_hip() -> bool:
    """True on ROCm builds of torch. Kept import-light: torch may not be loaded yet."""
    try:
        import torch

        return torch.version.hip is not None
    except Exception:
        return False


IS_HIP = _is_hip()

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
# On ROCm, `csrc/include_hip` supplies shims for the CUDA-only headers the
# kernels include (<cuda_bf16.h>, <cuda_fp16.h>, <cuda_runtime.h>, ...). It is
# appended only under IS_HIP so it can never shadow the real headers on CUDA.
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")] + (
    [str(KERNEL_PATH / "include_hip")] if IS_HIP else []
)
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
# `--expt-relaxed-constexpr` is nvcc-only; hipcc/clang++ rejects it outright.
# tvm_ffi already selects hipcc + `--offload-arch` on its own for ROCm, so the
# device flags are the only thing that needs branching here.
DEFAULT_CUDA_CFLAGS = (
    ["-std=c++20", "-O3"]
    if IS_HIP
    else ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
)
DEFAULT_LDFLAGS = []
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]

# nvcc-only device flags -> hipcc/clang equivalent. `None` means "drop it".
# Call sites pass these verbatim (e.g. extra_cuda_cflags=["--use_fast_math"]),
# so translate centrally instead of editing every caller.
_NVCC_TO_HIP_FLAGS: dict[str, str | None] = {
    "--use_fast_math": "-ffast-math",
    "-use_fast_math": "-ffast-math",
    "--expt-relaxed-constexpr": None,
    "--expt-extended-lambda": None,
    "--extended-lambda": None,
    "--ptxas-options=-v": None,
    "-Xptxas": None,
    "--generate-line-info": "-g",
    "-lineinfo": "-g",
}


def translate_cuda_cflags(flags: List[str]) -> List[str]:
    """Map nvcc device flags to hipcc on ROCm; pass through unchanged on CUDA."""
    if not IS_HIP:
        return list(flags)
    out: List[str] = []
    for flag in flags:
        if flag in _NVCC_TO_HIP_FLAGS:
            replacement = _NVCC_TO_HIP_FLAGS[flag]
            if replacement is not None:
                out.append(replacement)
            continue
        if flag.startswith(("-gencode", "--generate-code", "-arch=", "--gpu-architecture")):
            continue  # tvm_ffi supplies --offload-arch itself
        out.append(flag)
    return out


class CppArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


class KernelConfig(NamedTuple):
    num_threads: int
    max_occupancy: int
    use_pdl: bool


def _make_name(*args: str) -> str:
    return "minisgl__" + "_".join(str(arg) for arg in args)


def _make_wrapper(tup: Tuple[str, str]) -> str:
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):
            return "true" if arg else "false"
        if isinstance(arg, (int, float)):
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArgList(_convert(arg) for arg in args)


def load_aot(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    from tvm_ffi.cpp import load

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    return load(
        _make_name(*args),
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=translate_cuda_cflags(DEFAULT_CUDA_CFLAGS + extra_cuda_cflags),
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    from tvm_ffi.cpp import load_inline

    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    return load_inline(
        _make_name(*args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=translate_cuda_cflags(DEFAULT_CUDA_CFLAGS + extra_cuda_cflags),
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )
