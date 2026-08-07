from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import (
    destroy_distributed,
    enable_pynccl_distributed,
    get_tp_info,
    set_tp_info,
)
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


def resolve_kv_dtype(spec: str, compute_dtype: torch.dtype) -> Tuple[torch.dtype, torch.dtype]:
    """KV cache storage dtype from the `kv_cache_dtype` spec.

    gfx942/CDNA3's native FP8 is the **fnuz** variant (no negative zero, exponent bias 8), so
    that is what "fp8" means here -- `e4m3fn` is accepted explicitly for checkpoint-compatible
    storage, but fnuz is the type the hardware's MFMA path would want if attention is ever
    taught to consume fp8 directly instead of dequantising on load.

    No scale factor is applied. That is deliberate and it is specific to FP8 being a *floating
    point* format: e4m3 carries 3 mantissa bits, so its relative precision (~6% typical, 12.5%
    worst case) is the same anywhere inside its range. Scaling would buy nothing unless values
    clip at +-240 or fall under the 0.0078 smallest-normal -- which is why
    gptoss_fp8_kv_parity.py measures the real K/V range rather than assuming.
    """
    e4m3 = torch.float8_e4m3fnuz if torch.version.hip is not None else torch.float8_e4m3fn
    if spec == "auto":
        return compute_dtype, compute_dtype
    if spec in ("fp8", "fp8_e4m3"):
        return e4m3, e4m3
    if spec == "fp8_v":
        # V only. K stays at compute precision because its error is exponentiated by the
        # softmax while V's is not -- fp8 on both fails T-10 with 5 real divergences, two at
        # 20+ logit margins (21_fp8_kv_cache.md §2). Saves 25% of KV bytes instead of 50%.
        return compute_dtype, e4m3
    if spec == "fp8_e4m3fn":
        return torch.float8_e4m3fn, torch.float8_e4m3fn
    if spec == "fp8_e5m2":
        # More exponent range, 2 mantissa bits. Only sensible if the range check shows clipping.
        d = torch.float8_e5m2fnuz if torch.version.hip is not None else torch.float8_e5m2
        return d, d
    raise ValueError(
        f"unsupported kv_cache_dtype {spec!r}; expected auto, fp8, fp8_v, fp8_e4m3fn or fp8_e5m2"
    )


try:
    from minisgl.distributed import get_tp_info
    _HAS_TP_INFO = True
except Exception:  # noqa: BLE001
    _HAS_TP_INFO = False

    def get_tp_info():  # type: ignore[misc]
        raise RuntimeError


@contextlib.contextmanager
def _maybe_profile(batch: Batch, device: torch.device):
    """Thin wrapper over minisgl.profiling.profile_region.

    The implementation moved to minisgl/profiling.py so the AFD workers can share it -- they had
    no profiling at all, which is how doc 26 §3 came to attribute a step's cost by subtraction.
    Keeping two copies would guarantee the two hard-won lessons (the DeviceType.CUDA filter and
    TIME_ONLY) drift apart.
    """
    from minisgl.profiling import profile_region

    with profile_region(
        tag="engine",
        phase=batch.phase,
        n_tok=int(batch.input_ids.numel()),
        device=device,
        rank=get_tp_info().rank if _HAS_TP_INFO else 0,
    ):
        yield

def _init_tp_communication(config: EngineConfig, dtype: torch.dtype) -> torch.distributed.ProcessGroup:
    """Initialize tensor-parallel process group. Shared by local and afd engines."""
    from datetime import timedelta

    rank_override = getattr(config, "distributed_rank", None)
    world_size_override = getattr(config, "distributed_world_size", None)
    rank = int(config.tp_info.rank if rank_override is None else rank_override)
    world_size = int(config.tp_info.size if world_size_override is None else world_size_override)
    tp_groups = getattr(config, "distributed_tp_groups", None)
    tp_group_ranks = (
        tuple(int(r) for r in tp_groups[int(getattr(config, "dp_rank", 0))])
        if tp_groups is not None
        else tuple(range(config.tp_info.size))
    )

    if config.tp_info.size == 1 or config.use_pynccl:
        torch.distributed.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=config.distributed_timeout),
            init_method=config.distributed_addr,
        )
        if world_size == config.tp_info.size:
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
        else:
            groups = tp_groups or (tp_group_ranks,)
            tp_cpu_group = None
            for group_ranks in groups:
                group = torch.distributed.new_group(
                    ranks=[int(r) for r in group_ranks],
                    backend="gloo",
                )
                if rank in set(int(r) for r in group_ranks):
                    tp_cpu_group = group
            if tp_cpu_group is None:
                raise RuntimeError(
                    "Current distributed rank is not in any TP group: "
                    f"rank={rank} tp_groups={groups}"
                )
        max_bytes = config.max_forward_len * config.model_config.hidden_size * dtype.itemsize
        enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=config.distributed_timeout),
            init_method=config.distributed_addr,
        )
        if world_size == config.tp_info.size:
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
        else:
            groups = tp_groups or (tp_group_ranks,)
            tp_cpu_group = None
            for group_ranks in groups:
                group = torch.distributed.new_group(
                    ranks=[int(r) for r in group_ranks],
                    backend="gloo",
                )
                if rank in set(int(r) for r in group_ranks):
                    tp_cpu_group = group
            if tp_cpu_group is None:
                raise RuntimeError(
                    "Current distributed rank is not in any TP group: "
                    f"rank={rank} tp_groups={groups}"
                )
        assert tp_cpu_group is not None
    return tp_cpu_group


_NSYS_ENV_PREFIXES = (
    "NSYS_",
    "QUADD_",
)

_NSYS_ENV_KEYS = (
    "CUDA_INJECTION64_PATH",
    "CUPTI_PROFILE_MODE",
    "NVTX_INJECTION64_PATH",
    "TRACE_START_IMMEDIATELY",
    "USE_AGENT_API",
    "__GL_CONSTANT_FRAME_RATE_HINT",
)

_NSYS_LD_PRELOAD_MARKERS = (
    "nsight-systems",
    "libToolsInjection",
    "libLinuxKeyboardInterceptorProxy",
)


def _build_nsys_sanitized_env(env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if env is None else env)

    for key in list(env):
        if key in _NSYS_ENV_KEYS or any(key.startswith(prefix) for prefix in _NSYS_ENV_PREFIXES):
            env.pop(key, None)

    ld_preload = env.get("LD_PRELOAD", "")
    if ld_preload:
        kept_entries = []
        for entry in ld_preload.split(":"):
            if not entry:
                continue
            if any(marker in entry for marker in _NSYS_LD_PRELOAD_MARKERS):
                continue
            else:
                kept_entries.append(entry)
        if kept_entries:
            env["LD_PRELOAD"] = ":".join(kept_entries)
        else:
            env.pop("LD_PRELOAD", None)

    return env


def _install_flashinfer_ninja_env_sanitizer() -> None:
    try:
        import flashinfer.jit.core as flashinfer_jit_core
        import flashinfer.jit.cpp_ext as flashinfer_cpp_ext
    except ImportError:
        return

    if getattr(flashinfer_cpp_ext, "_minisgl_nsys_run_ninja_patched", False):
        return

    def _sanitized_run_ninja(workdir, ninja_file, verbose):
        workdir.mkdir(parents=True, exist_ok=True)
        command = [
            "ninja",
            "-v",
            "-C",
            str(workdir.resolve()),
            "-f",
            str(ninja_file.resolve()),
        ]
        num_workers = flashinfer_cpp_ext._get_num_workers()
        if num_workers is not None:
            command += ["-j", str(num_workers)]

        sys.stdout.flush()
        sys.stderr.flush()
        try:
            subprocess.run(
                command,
                stdout=None if verbose else subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(workdir.resolve()),
                check=True,
                text=True,
                env=_build_nsys_sanitized_env(),
            )
        except subprocess.CalledProcessError as e:
            msg = "Ninja build failed."
            if e.output:
                msg += " Ninja output:\n" + e.output
            raise RuntimeError(msg) from e

    flashinfer_cpp_ext.run_ninja = _sanitized_run_ninja
    flashinfer_jit_core.run_ninja = _sanitized_run_ninja
    flashinfer_cpp_ext._minisgl_nsys_run_ninja_patched = True
    logger.info("Installed FlashInfer ninja env sanitizer for Nsight profiling")


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event


class Engine:
    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        _adjust_config(config)

        visible_device_count = torch.cuda.device_count()
        device_index = int(os.environ.get("MINISGL_DEVICE_INDEX", config.tp_info.rank))
        if device_index >= visible_device_count:
            # Ray-per-rank actors typically expose a single visible GPU even for rank > 0.
            device_index = 0
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        torch.manual_seed(42)
        self.stream = torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        self.dtype = config.dtype
        self.kv_dtype, self.v_dtype = resolve_kv_dtype(config.kv_cache_dtype, self.dtype)
        self.ctx = Context(config.page_size)
        set_global_ctx(self.ctx)
        self._ray_nsys_enabled = bool(getattr(config, "ray_nsys", False))
        if self._ray_nsys_enabled:
            _install_flashinfer_ninja_env_sanitizer()

        try:
            logger.info(
                "Engine init start: device=%s tp_rank=%d/%d attn=%s moe=%s graph_max_bs=%s",
                self.device,
                config.tp_info.rank,
                config.tp_info.size - 1,
                config.attention_backend,
                config.moe_backend,
                config.cuda_graph_max_bs,
            )
            self.tp_cpu_group = _init_tp_communication(config, self.dtype)
            init_free_memory = self._sync_get_memory()[1]
            logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

            # ======================= Model initialization ========================
            set_rope_device(self.device)
            with torch.device("meta"), torch_dtype(config.dtype):
                self.model = create_model(config.model_config)
            self.model.load_state_dict(self._load_weight_state_dict(config))

            # ======================= KV cache initialization ========================
            self.num_pages = self._determine_num_pages(init_free_memory, config)
            num_tokens = self.num_pages * config.page_size
            self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
                model_config=config.model_config,
                num_pages=self.num_pages + 1,  # +1 for dummy page
                page_size=config.page_size,
                device=self.device,
                dtype=self.kv_dtype,
                v_dtype=self.v_dtype,
            )
            if (self.kv_dtype, self.v_dtype) != (self.dtype, self.dtype):
                ratio = (2 * self.dtype.itemsize) / (
                    self.kv_dtype.itemsize + self.v_dtype.itemsize
                )
                logger.info_rank0(
                    "KV cache dtype: K=%s V=%s (compute %s) -- %.2fx the pages for the "
                    "same memory", self.kv_dtype, self.v_dtype, self.dtype, ratio,
                )

            # ======================= Page table initialization ========================
            # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
            self.max_seq_len = min(config.max_seq_len, num_tokens)
            aligned_max_seq_len = _align_up_32(self.max_seq_len)
            self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
                (config.max_running_req + 1, aligned_max_seq_len),
                dtype=torch.int32,
                device=self.device,
            )

            # ======================= Attention & MoE backend initialization ========================
            self.ctx.attn_backend = self.attn_backend = create_attention_backend(
                config.attention_backend, config.model_config
            )
            if config.model_config.is_moe:
                self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

            # ======================= Sampler initialization ========================
            self.sampler = Sampler(self.device, config.model_config.vocab_size)

            post_free_memory = self._sync_get_memory()[0]
            logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

            # ======================= Graph capture initialization ========================
            self.dummy_req = Req(
                input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
                table_idx=config.max_running_req,
                cached_len=0,
                output_len=1,
                uid=-1,
                sampling_params=None,  # type: ignore
                cache_handle=None,  # type: ignore
            )
            self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
            self.graph_runner = GraphRunner(
                stream=self.stream,
                device=self.device,
                model=self.model,
                attn_backend=self.attn_backend,
                cuda_graph_bs=config.cuda_graph_bs,
                cuda_graph_max_bs=config.cuda_graph_max_bs,
                free_memory=init_free_memory,
                max_seq_len=aligned_max_seq_len,
                vocab_size=config.model_config.vocab_size,
                dummy_req=self.dummy_req,
            )
            logger.info("Engine init complete")
        finally:
            pass

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            is_minimax_m2 = (
                getattr(config.model_config, "architectures", [""])[0]
                == "MiniMaxM2ForCausalLM"
            )
            is_glm4_moe = (
                getattr(config.model_config, "architectures", [""])[0]
                == "Glm4MoeForCausalLM"
            )

            def _minimax_fp32_key(key: str) -> bool:
                return is_minimax_m2 and (
                    key.endswith(".mlp.gate.weight")
                    or key.endswith(".mlp.e_score_correction_bias")
                )

            def _glm4_fp32_key(key: str) -> bool:
                return is_glm4_moe and (
                    key.endswith(".mlp.gate.weight")
                    or key.endswith(".mlp.gate.e_score_correction_bias")
                )

            return {
                k: (
                    v
                    if v.dtype == torch.float8_e4m3fn
                    # Packed MXFP4 blocks/scales are uint8 BIT CONTAINERS, not numbers. Casting
                    # them to bf16 would silently turn every packed nibble pair into a float and
                    # destroy the weights -- no shape error, just garbage.
                    or v.dtype == torch.uint8
                    or (v.dtype == torch.float32 and k.endswith("_scale"))
                    else v.to(torch.float32)
                    if _minimax_fp32_key(k) or _glm4_fp32_key(k)
                    else v.to(self.dtype)
                )
                for k, v in load_weight(config.model_path, self.device)
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * config.model_config.num_layers
        ) // 2 * (self.kv_dtype.itemsize + self.v_dtype.itemsize)  # K and V sized separately
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        assert torch.cuda.current_stream() == self.stream
        with self.ctx.forward_batch(batch), _maybe_profile(batch, self.device):
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)
            else:
                logits = self.model.forward()

        for req in batch.reqs:
            req.complete_one()

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        copy_done_event = torch.cuda.Event()
        copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        if torch.version.hip is not None:
            # Both `trtllm` and `fi` are NVIDIA-only (TensorRT-LLM / FlashInfer,
            # neither of which builds on ROCm). Without this branch `auto`
            # resolved to "fi" on gfx942 and died importing flashinfer.
            #
            # Hybrid: `triton_prefill` prefills and `triton_decode` decodes. Split because
            # only the decode half is HIP-graph capturable (capture is worth 4.1x at
            # bs=1, dev_log/16) while prefill needs none -- GraphRunner only captures
            # decode. Prefill is worth 2.3-2.5x on TTFT (dev_log/24). `torch_ref` stays
            # registered as the correctness reference for both halves.
            backend = "triton_prefill,triton_decode"
        else:
            backend = "trtllm" if is_sm100_supported() else "fi"
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
