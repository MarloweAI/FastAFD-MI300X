from __future__ import annotations

import os
import shutil
from pathlib import Path


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def find_shared_lib(lib_dir: Path, stem: str) -> Path | None:
    direct = lib_dir / f"{stem}.so"
    if direct.exists():
        return direct
    matches = sorted(lib_dir.glob(f"{stem}.so.*"))
    return matches[0] if matches else None


def resolve_nvcc() -> str:
    if path := os.environ.get("CUDA_NVCC_EXECUTABLE"):
        return path
    if conda_prefix := os.environ.get("CONDA_PREFIX"):
        conda_nvcc = Path(conda_prefix) / "bin" / "nvcc"
        if conda_nvcc.exists():
            return str(conda_nvcc)
    if path := shutil.which("nvcc"):
        return path
    raise RuntimeError("Failed to find nvcc. Set CUDA_NVCC_EXECUTABLE or put nvcc on PATH.")


def resolve_cuda_lib_dir(nvcc_path: str) -> Path:
    roots: list[Path] = [Path(nvcc_path).resolve().parent.parent]
    if conda_prefix := os.environ.get("CONDA_PREFIX"):
        roots.append(Path(conda_prefix))
    if cuda_home := os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        roots.append(Path(cuda_home))
    roots = dedupe_paths(roots)

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(sorted(root.glob("targets/*/lib")))
        candidates.extend([root / "lib64", root / "lib"])
    for lib_dir in dedupe_paths(candidates):
        if not lib_dir.exists():
            continue
        if (lib_dir / "libcudadevrt.a").exists() and find_shared_lib(lib_dir, "libcudart"):
            return lib_dir
    searched = ", ".join(str(path) for path in dedupe_paths(candidates)) or "<none>"
    raise RuntimeError(f"Failed to locate CUDA runtime/device libs. Searched: {searched}")


_NSYS_ENV_PREFIXES = (
    "NSYS_",
    "QUADD_",
)

_NSYS_ENV_KEYS = (
    "CUDA_INJECTION64_PATH",
    "CUDA_INJECTION32_PATH",
    "CUPTI_PROFILE_MODE",
    "NVTX_INJECTION64_PATH",
    "NVTX_INJECTION32_PATH",
    "TRACE_START_IMMEDIATELY",
    "USE_AGENT_API",
    "__GL_CONSTANT_FRAME_RATE_HINT",
)

_NSYS_LD_PRELOAD_MARKERS = (
    "nsight",
    "nsys",
    "injection",
    "cupti",
    "quadd",
)


def sanitize_nsys_subprocess_env(env: dict[str, str]) -> dict[str, str]:
    """Remove Nsight injection only from child process environments.

    The profiled worker process must keep NVTX injection loaded so NVTX ranges
    appear in NSYS. Build helpers, however, spawn nvcc/cuobjdump children that
    can fail if they inherit the profiler injection libraries.
    """
    env = dict(env)
    for key in list(env):
        if key in _NSYS_ENV_KEYS or any(key.startswith(prefix) for prefix in _NSYS_ENV_PREFIXES):
            env.pop(key, None)

    ld_preload = env.get("LD_PRELOAD", "")
    if ld_preload:
        kept_entries = [
            entry
            for entry in ld_preload.split(":")
            if entry and not any(marker in entry.lower() for marker in _NSYS_LD_PRELOAD_MARKERS)
        ]
        if kept_entries:
            env["LD_PRELOAD"] = ":".join(kept_entries)
        else:
            env.pop("LD_PRELOAD", None)
    return env


def nvcc_build_env(nvcc_path: str) -> dict[str, str]:
    env = sanitize_nsys_subprocess_env(dict(os.environ))
    for key in (
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
    ):
        env.pop(key, None)
    nvcc_root = Path(nvcc_path).resolve().parent.parent
    env["CUDA_HOME"] = str(nvcc_root)
    env["CUDA_PATH"] = str(nvcc_root)
    return env
