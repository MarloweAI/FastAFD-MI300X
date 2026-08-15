"""
Shared AFD Ray worker runtime.

BaseAfdWorker owns the common per-GPU runtime used by AFD attention/model actors.
"""
from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any, Literal

import torch
from minisgl.distributed import (
    DistributedInfo,
    set_ep_info,
    set_tp_info,
    try_get_ep_info,
    try_get_tp_info,
)
from minisgl.env import afd_env
from minisgl.message import BaseTokenizerMsg
from minisgl.utils import (
    ZmqPullQueue,
    ZmqPushQueue,
    div_even,
    init_nvtx_cpu_trace,
    nvtx_label,
    nvtx_range,
    shutdown_nvtx_cpu_trace,
)

from .afd_protocol import (
    AfdCommand,
    AfdProfilerCmd,
    AfdReply,
    AfdTopology,
    AfdWorkerReadyReply,
)
from .afd_support import AfdRuntimeConfig, flush_log_lines, log_line

__all__ = ["BaseAfdWorker"]


def ensure_afd_parallel_info(config: Any) -> None:
    """Install TP/EP rank metadata for the current AFD worker process."""
    tp_info = try_get_tp_info()
    if tp_info is None:
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
    elif tp_info.rank != config.tp_info.rank or tp_info.size != config.tp_info.size:
        raise RuntimeError(
            f"TP info is already set to {tp_info.rank}/{tp_info.size}, "
            f"cannot reset to {config.tp_info.rank}/{config.tp_info.size}"
        )

    ep_size = int(getattr(config, "ep_size", 1))
    if ep_size <= 1:
        return
    per_replica_size = config.tp_info.size
    cross_replica_size = max(1, int(getattr(config, "dp_size", 1))) * config.tp_info.size
    if ep_size not in (per_replica_size, cross_replica_size):
        raise RuntimeError(
            "AFD MoE currently supports EP only as per-MLP-DP EP or full-world "
            "EP, so ep_size must be in "
            f"{{tp_size={per_replica_size}, dp_size*tp_size={cross_replica_size}}} "
            "when EP is enabled. Got "
            f"ep_size={ep_size} (dp_size={getattr(config, 'dp_size', 1)})"
        )
    if ep_size == cross_replica_size and ep_size != per_replica_size:
        ep_rank = int(getattr(config, "dp_rank", 0)) * config.tp_info.size + config.tp_info.rank
    else:
        ep_rank = config.tp_info.rank
    ep_info = try_get_ep_info()
    if ep_info is None:
        set_ep_info(rank=ep_rank, size=ep_size)
        return
    if ep_info.rank != ep_rank or ep_info.size != ep_size:
        raise RuntimeError(
            f"EP info is already set to {ep_info.rank}/{ep_info.size}, "
            f"cannot reset to {ep_rank}/{ep_size}"
        )


class BaseAfdWorker:
    """Shared per-GPU worker runtime for AFD attention/model actors."""

    def __init__(
        self,
        *,
        role: Literal["attention", "mlp"],
        tp_rank: int,
        tp_size: int,
        comm_rank: int,
        node_ip: str,
        distributed_host: str,
        distributed_port: int,
        model_path: str,
        attention_backend: str,
        max_seq_len: int,
        max_comm_tokens: int,
        max_extend_tokens: int | None,
        max_running_req: int,
        num_page_override: int | None,
        cuda_graph_max_bs: int,
        decode_graph_bs: tuple[int, ...],
        enable_attention_decode_graph: bool,
        enable_model_decode_graph: bool,
        page_size: int,
        cache_type: str,
        log_dir: str,
        use_dummy_weight: bool,
        device_comm_num_sms: int,
        ep_size: int = 1,
        moe_a2a_backend: str = "none",
        moe_runner_backend: str = "auto",
        ray_nsys: bool = False,
        disable_overlap: bool = False,
        afd_num_mb: int = 1,
        rpc_timeout_ms: int = 300_000,
        attn_dp_rank: int = 0,
        attn_dp_size: int = 1,
        mlp_dp_rank: int = 0,
        mlp_dp_size: int = 1,
        attn_tp_size: int | None = None,
        mlp_tp_size: int | None = None,
        memory_ratio: float = 0.8,
        distributed_rank: int | None = None,
        distributed_world_size: int | None = None,
        distributed_tp_groups: tuple[tuple[int, ...], ...] | None = None,
    ):
        import ray.util
        self.role = role
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.attn_dp_rank = int(attn_dp_rank)
        self.attn_dp_size = max(1, int(attn_dp_size))
        self.mlp_dp_rank = int(mlp_dp_rank)
        self.mlp_dp_size = max(1, int(mlp_dp_size))
        self.attn_tp_size = int(attn_tp_size if attn_tp_size is not None else tp_size)
        self.mlp_tp_size = int(mlp_tp_size if mlp_tp_size is not None else tp_size)
        self._topology = AfdTopology(
            attn_dp_size=self.attn_dp_size,
            mlp_dp_size=self.mlp_dp_size,
            attn_tp_size=self.attn_tp_size,
            mlp_tp_size=self.mlp_tp_size,
            ep_size=int(ep_size),
        )
        self.comm_rank = comm_rank
        self.worker_rank = comm_rank
        self.node_ip = ray.util.get_node_ip_address()
        assert self.node_ip == node_ip, (self.node_ip, node_ip)
        if self.role == "mlp" and self.mlp_dp_size > 1:
            self.log_path = str(Path(log_dir) / f"{role}_dp{self.mlp_dp_rank}_rank{tp_rank}.log")
        elif self.attn_dp_size > 1:
            self.log_path = str(Path(log_dir) / f"{role}_dp{self.attn_dp_rank}_rank{tp_rank}.log")
        else:
            self.log_path = str(Path(log_dir) / f"{role}_rank{tp_rank}.log")
        ray_gpu_ids = tuple(ray.get_gpu_ids())
        initial_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        # Logging only CUDA_VISIBLE_DEVICES is what made the ROCm binding bug invisible:
        # the banner read `initial_cvd=None effective_cvd=None` and looked correct while
        # HIP_VISIBLE_DEVICES was the variable actually restricting the actor.
        initial_hvd = os.environ.get("HIP_VISIBLE_DEVICES")
        use_megamoe_device_binding = afd_env("MOE_BACKEND", "deepep") == "megamoe_m2n"
        device_index = 0
        if use_megamoe_device_binding:
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "megamoe_m2n requires the GPU visibility variables to be cleared "
                    "before CUDA is initialized"
                )
            if ray_gpu_ids:
                device_index = int(ray_gpu_ids[0])
            # torch symmetric memory exchanges device ordinals.  If Ray remaps each
            # actor to a single visible cuda:0 device, same-node ranks look overlapping
            # even when they are on different physical GPUs.  So widen visibility from
            # this actor's single card back to the driver's slice, and bind to Ray's
            # index -- which is numbered WITHIN that slice, so it is only a valid
            # ordinal once the slice is restored.
            #
            # Popping CUDA_VISIBLE_DEVICES alone is a NO-OP on ROCm: the variable that
            # gates HIP visibility is HIP_VISIBLE_DEVICES. That is why megamoe_m2n died
            # with "HIP error: invalid device ordinal" at set_device for every expert
            # rank (ray id 2 or 3 against a single visible card) while attention rank 0
            # happened to survive on ray id 0. doc 26 §2.4.
            #
            # Restoring the driver's slice rather than clearing outright also keeps the
            # shared-host card restriction intact. Clearing would expose all 8 cards and
            # make Ray's index a *global* ordinal, which coincides with the intended card
            # only when GPUS is a 0-based contiguous prefix -- the same identity-map trap
            # that hid the ROCR/HIP composition bug (run_afd_rocm.sh:86). With GPUS=4,5,6,7
            # it would silently bind another user's GPU instead of failing.
            host_slice = os.environ.get("MINISGL_AFD_HOST_VISIBLE_DEVICES")
            for _var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
                if host_slice:
                    os.environ[_var] = host_slice
                else:
                    os.environ.pop(_var, None)
            if host_slice is not None:
                n_visible = len([p for p in host_slice.split(",") if p.strip() != ""])
                if device_index >= n_visible:
                    raise RuntimeError(
                        f"megamoe_m2n device binding out of range: ray_gpu_ids="
                        f"{ray_gpu_ids} -> index {device_index}, but the driver slice "
                        f"MINISGL_AFD_HOST_VISIBLE_DEVICES={host_slice!r} exposes only "
                        f"{n_visible} card(s). Refusing rather than binding the wrong GPU."
                    )
            os.environ["MINISGL_DEVICE_INDEX"] = str(device_index)
        self.device = torch.device(f"cuda:{device_index}")
        torch.cuda.set_device(self.device)
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] cuda_binding "
            f"backend={afd_env('MOE_BACKEND', 'deepep')} "
            f"initial_cvd={initial_cvd!r} effective_cvd={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
            f"initial_hvd={initial_hvd!r} effective_hvd={os.environ.get('HIP_VISIBLE_DEVICES')!r} "
            f"host_slice={os.environ.get('MINISGL_AFD_HOST_VISIBLE_DEVICES')!r} "
            f"ray_gpu_ids={ray_gpu_ids} device={self.device} "
            f"device_count={torch.cuda.device_count()}",
            flush=True,
        )
        self.dtype = torch.bfloat16
        self.attention_backend = str(attention_backend)
        self.page_size = int(page_size)
        self.max_seq_len = int(max_seq_len)
        self.max_comm_tokens = int(max_comm_tokens)
        self.max_running_req = int(max_running_req)
        self.cuda_graph_max_bs = int(cuda_graph_max_bs)
        self.decode_graph_bs = tuple(sorted(int(b) for b in decode_graph_bs))
        self.enable_attention_decode_graph = bool(enable_attention_decode_graph)
        self.enable_model_decode_graph = bool(enable_model_decode_graph)
        # AG and EG decode graphs contain matching collective calls, so capture and
        # replay must be enabled on both sides or disabled on both sides.
        self.enable_decode_graph = (
            self.enable_attention_decode_graph and self.enable_model_decode_graph
        )
        self.device_comm_num_sms = max(1, int(device_comm_num_sms))
        self.mlp_ep_size = int(ep_size)
        self.ep_size = self.mlp_ep_size if role == "mlp" else 1
        self.disable_overlap = bool(disable_overlap)
        self.afd_num_mb = max(1, int(afd_num_mb))
        self._control_profile_start_step = int(
            os.environ.get("MINISGL_AFD_CONTROL_PROFILE_START_STEP", "0") or 0
        )
        self._control_profile_stop_step = int(
            os.environ.get("MINISGL_AFD_CONTROL_PROFILE_STOP_STEP", "0") or 0
        )
        self._recv_timeout_ms = max(1_000, int(rpc_timeout_ms))
        self._dual_stream_microbatch = self.afd_num_mb > 1
        self._comm_stream = torch.cuda.Stream(device=self.device)
        self._afd_comm_streams = [
            torch.cuda.Stream(device=self.device) for _ in range(max(1, self.afd_num_mb))
        ]
        # Dedicated D2H stream for the one-step-ahead (overlap) AG/EG loops: the
        # sampled next_tokens copy runs here async, its copy_done_event is
        # synchronized one step LATER (mirrors the one-step-ahead EG worker), so the
        # compute stream never stalls on a gpu->cpu sync between steps.
        self._d2h_stream = torch.cuda.Stream(device=self.device)
        self._comm_ready = False
        self._recv_from_coordinator: ZmqPullQueue[AfdCommand] | None = None
        self._send_to_coordinator: ZmqPushQueue[AfdReply] | None = None
        self._send_into_detokenizer: ZmqPushQueue[BaseTokenizerMsg] | None = None
        self._hot_loop_running = False
        self._ray_nsys_enabled = bool(ray_nsys)
        self._nsys_runtime_started = False
        if self.role == "mlp" and self.mlp_dp_size > 1:
            trace_label = f"{role}_dp{self.mlp_dp_rank}_rank{tp_rank}"
        elif self.attn_dp_size > 1:
            trace_label = f"{role}_dp{self.attn_dp_rank}_rank{tp_rank}"
        else:
            trace_label = f"{role}_rank{tp_rank}"
        self._cpu_trace_path = init_nvtx_cpu_trace(
            output_dir=log_dir,
            process_label=trace_label,
            enabled=bool(ray_nsys),
            metadata={
                "kind": "afd-worker",
                "role": role,
                "attn_dp_rank": self.attn_dp_rank,
                "attn_dp_size": self.attn_dp_size,
                "mlp_dp_rank": self.mlp_dp_rank,
                "mlp_dp_size": self.mlp_dp_size,
                "tp_rank": tp_rank,
                "tp_size": tp_size,
                "comm_rank": comm_rank,
                "node_ip": node_ip,
                "ray_nsys": bool(ray_nsys),
            },
        )
        if self._cpu_trace_path is not None:
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] cpu_trace enabled path={self._cpu_trace_path}",
                flush=True,
            )

        config = AfdRuntimeConfig(
            model_path=model_path,
            tp_info=DistributedInfo(tp_rank, tp_size),
            dtype=torch.bfloat16,
            max_running_req=max_running_req,
            attention_backend=attention_backend,
            moe_backend="fused",
            cuda_graph_max_bs=self.cuda_graph_max_bs,
            page_size=self.page_size,
            memory_ratio=float(memory_ratio),
            use_dummy_weight=use_dummy_weight,
            max_extend_tokens=(
                int(max_extend_tokens)
                if max_extend_tokens is not None
                else AfdRuntimeConfig.max_extend_tokens
            ),
            max_seq_len_override=max_seq_len,
            num_page_override=num_page_override,
            ep_size=self.ep_size,
            dp_size=(self.mlp_dp_size if role == "mlp" else self.attn_dp_size),
            dp_rank=(self.mlp_dp_rank if role == "mlp" else self.attn_dp_rank),
            cache_type=cache_type,
            offline_mode=True,
            distributed_host=distributed_host,
            distributed_port=distributed_port,
            distributed_rank=distributed_rank,
            distributed_world_size=distributed_world_size,
            distributed_tp_groups=distributed_tp_groups,
            ray_nsys=bool(ray_nsys),
            afd_num_mb=self.afd_num_mb,
            afd_attn_dp_size=self.attn_dp_size,
            afd_mlp_dp_size=self.mlp_dp_size,
            afd_attn_tp_size=self.attn_tp_size,
            afd_mlp_tp_size=self.mlp_tp_size,
            afd_moe_a2a_backend=str(moe_a2a_backend),
            afd_moe_runner_backend=str(moe_runner_backend),
        )
        self.runtime = self._create_runtime(config)
        self.runtime_state = getattr(self.runtime, "state", self.runtime)
        try:
            ctx = self.runtime_state.ctx
            ctx.server_args = config
            # The colocated engine sets ctx.moe_backend (engine.py:266); the AFD path
            # never did, because its only supported expert runner was deep_gemm, which
            # does not use it. TritonRunner.apply() does -- it calls
            # ctx.moe_backend.forward(...) -- so the EG side needs it populated or it
            # dies with AttributeError on the first expert launch.
            if getattr(ctx, "moe_backend", None) is None:
                from minisgl.moe import create_moe_backend

                ctx.moe_backend = create_moe_backend(str(config.moe_backend))
        except AttributeError:
            pass

        model_config = config.model_config
        self.head_dim = model_config.head_dim
        # Local view: heads/dims this worker's own TP rank holds.
        #
        # Only the attention role shards attention heads. This used to divide by
        # `tp_size` for BOTH roles, so an MLP pool whose size did not divide
        # num_qo_heads asserted at construction -- e.g. 1A+3F on 4 cards died with
        # "a = 32 must be divisible by b = 3" from the *expert* worker. These three
        # fields are write-only (nothing in the tree reads them, upstream included),
        # so that assert was rejecting valid A/F splits for no reason. Keep the
        # fields, but compute them only where they mean something.
        # See dev_log/qwen/18_afd_split_sweep.md.
        if self.role == "attention":
            self.local_qo_heads = div_even(model_config.num_qo_heads, tp_size)
            self.local_qo_dim = self.local_qo_heads * self.head_dim
            self.local_kv_dim = (
                div_even(model_config.num_kv_heads, tp_size, allow_replicate=True)
                * self.head_dim
            )
        else:
            # The MLP pool holds no attention heads; it is sized by hidden_size.
            self.local_qo_heads = 0
            self.local_qo_dim = 0
            self.local_kv_dim = 0
        self.num_layers = int(model_config.num_layers)
        self.hidden_size = int(model_config.hidden_size)
        self.vocab_size = int(model_config.vocab_size)
        self.num_experts = int(getattr(model_config, "num_experts", 0) or 0)
        self.top_k = int(getattr(model_config, "num_experts_per_tok", 0) or 0)
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] runtime_ready node_ip={self.node_ip} "
            f"comm_rank={self.comm_rank} "
            f"attn_dp_rank={self.attn_dp_rank} attn_dp_size={self.attn_dp_size} "
            f"mlp_dp_rank={self.mlp_dp_rank} mlp_dp_size={self.mlp_dp_size} "
            f"attn_tp_size={self.attn_tp_size} mlp_tp_size={self.mlp_tp_size} "
            f"device_comm_num_sms={self.device_comm_num_sms} "
            f"afd_num_mb={self.afd_num_mb} "
            f"dual_stream_microbatch={self._dual_stream_microbatch} "
            f"rpc_timeout_ms={self._recv_timeout_ms} "
            f"decode_graph_bs={list(self.decode_graph_bs)} "
            f"attention_decode_graph={self.enable_attention_decode_graph} "
            f"model_decode_graph={self.enable_model_decode_graph}",
            flush=True,
        )

    def _create_runtime(self, config: AfdRuntimeConfig):
        raise NotImplementedError

    def _control_profile_enabled(self, step_id: int) -> bool:
        start = int(self._control_profile_start_step)
        stop = int(self._control_profile_stop_step)
        if start <= 0 and stop <= 0:
            return False
        step_id = int(step_id)
        return (start <= 0 or step_id >= start) and (stop <= 0 or step_id <= stop)

    @staticmethod
    def _deepep_do_cpu_sync(phase: object) -> bool:
        if os.environ.get("MINISGL_AFD_DISABLE_DEEPEP_CPU_SYNC", "0") == "1":
            return False
        return str(phase) != "decode"

    def _run_hot_rpc_loop(self) -> None:
        raise NotImplementedError

    def _prepare_hot_rpc_runtime(self) -> None:
        # nsys-safe JIT: child cuobjdump/nvcc must not inherit nsys CUDA
        # injection before DeepEP/DeepGEMM JIT compilation starts.
        self._detach_nsys_injection_from_subprocesses()
        self._init_afd_runtime()
        # AG and EG graph capture contains matching collectives, so all ranks
        # self-trigger warmup here before the coordinator starts sending steps.
        self.warmup_afd_decode_graphs(self.decode_graph_bs)
        # Route graphs are collective-free, so unlike the whole-step graphs above they need no
        # cross-rank lockstep -- but they DO need a quiescent stream, which is only true here,
        # before the coordinator starts sending steps.
        if self._ag_route_graphs is not None:
            self._ag_route_graphs.warmup()
        if self._ag_attn_graphs is not None:
            # Needs the runtime's dummy req to build a capture batch, and the same bucket list
            # as route graphs so a decode step either finds both graphs or neither.
            _raw = os.environ.get("MINISGL_AFD_AG_ROUTE_GRAPH_BS", "1,2,4,8,16,32")
            try:
                _bs = tuple(sorted({int(x) for x in _raw.split(",") if x.strip()}))
            except ValueError:
                _bs = (1, 2, 4, 8, 16, 32)
            self._ag_attn_graphs.warmup(
                bs_list=_bs, dummy_req=getattr(self.runtime, "_dummy_req", None)
            )
        if self._eg_expert_graphs is not None:
            # Same quiescent-point requirement as the AG graphs: collective-free to capture, but it
            # needs a stream with nothing in flight, which is only true here, before the
            # coordinator starts sending steps.
            self._eg_expert_graphs.warmup()
        self._warmup_fused_plan()
        self._send_hot_loop_ready()
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] hot_rpc_loop:ready_after_afd_init",
            flush=True,
        )

    def _recv_hot_command(self, *, blocking: bool) -> AfdCommand | None:
        queue = self._recv_from_coordinator
        if queue is None:
            raise RuntimeError("Hot RPC loop is not initialized")
        recv_start_ns = time.perf_counter_ns()
        if blocking:
            try:
                cmd = queue.get(timeout_ms=self._recv_timeout_ms)
            except TimeoutError:
                return None
        else:
            # true non-blocking (queue.get() with no timeout blocks on socket.recv)
            if queue.empty():
                return None
            cmd = queue.get()
        recv_done_ns = time.perf_counter_ns()
        self._log_plan_recv(
            cmd,
            blocking=blocking,
            recv_wait_ms=(recv_done_ns - recv_start_ns) / 1_000_000.0,
        )
        return cmd

    def _log_plan_recv(
        self,
        cmd: AfdCommand,
        *,
        blocking: bool,
        recv_wait_ms: float,
    ) -> None:
        pass

    def _ensure_comm_ready(self) -> None:
        if self._comm_ready:
            return
        self._comm_ready = True
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] afd runtime ready",
            flush=True,
        )

    def prepare_communication_runtime(self) -> dict[str, Any]:
        self._ensure_comm_ready()
        return {"role": self.role, "rank": self.tp_rank, "ready": self._comm_ready}

    def get_runtime_info(self) -> dict[str, Any]:
        info = {
            "role": self.role,
            "rank": self.tp_rank,
            "num_layers": int(self.num_layers),
        }
        if self.role == "attention":
            info["num_pages"] = int(self.runtime_state.num_pages)
            info["max_seq_len"] = int(self.runtime_state.max_seq_len)
        return info

    def _init_afd_runtime(self) -> None:
        """afd AG/EG: build the role model + role-split weights + the directional
        DeepEP adapter over the union world. Collective (all union ranks build the
        adapter together); idempotent; called once at hot-loop start."""
        if getattr(self, "_afd_initialized", False):
            return
        import torch.distributed as dist
        from minisgl.engine.sample import Sampler
        from minisgl.models import load_weight
        from minisgl.models.afd import extract_stage_state_dict
        from minisgl.utils import torch_dtype

        # Imported lazily inside the deepep branch below: on ROCm the DeepEP
        # extension cannot build, and the `rccl` backend must not depend on it
        # even transitively.
        DeepEPM2NAdapter = None  # noqa: N806  (resolved in the deepep branch)

        runtime_state = self.runtime_state
        config = runtime_state.config
        mc = config.model_config
        ag_size = self._topology.total_attn_workers
        eg_size = self._topology.total_mlp_workers
        # self.max_comm_tokens is sized conservatively for fan-in topologies.
        # DeepEP's num_max_dispatch_tokens_per_rank is per AG source rank, so
        # undo that expansion for the afd union adapter.
        dp_fanin = self._topology.attn_fanin_per_mlp_dp
        afd_max_dispatch_tokens = max(1, int(getattr(self, "max_comm_tokens", 0)) // dp_fanin)
        bucket = max(int(self.cuda_graph_max_bs), afd_max_dispatch_tokens)

        # Directional dispatch/combine over the union world (all AG+EG ranks).
        # Pipeline scheduling intentionally keeps multiple dispatch handles alive:
        # D(L,MB0), D(L,MB1), C(L,MB0), D(L+1,MB0), ...
        # A DeepEP ElasticBuffer owns the metadata/workspace that combine reads
        # through the dispatch handle, so each in-flight MB gets one adapter lane.
        adapter_lanes = max(1, int(self.afd_num_mb))
        arch0 = getattr(mc, "architectures", [""])[0]
        model_extra = getattr(mc, "model_extra", {})
        # On ROCm default to the `rccl` transport: DeepEP cannot build on gfx942
        # (NCCL device API + SM90 TMA), so `deepep` would fail at expert-worker init.
        # Overridable with MINISGL_AFD_MOE_BACKEND.
        _default_m2n = "rccl" if torch.version.hip is not None else "deepep"
        self._afd_moe_backend = afd_env("MOE_BACKEND", _default_m2n)
        if self._afd_moe_backend == "megamoe_m2n":
            os.environ.setdefault("MINISGL_MEGAMOE_AG_SMS", "24")
            n_shared_experts = int(model_extra.get("n_shared_experts", 0) or 0)
            remote_shared_experts = (
                arch0 == "Glm4MoeForCausalLM"
                and n_shared_experts > 0
            )
            if remote_shared_experts and n_shared_experts != 1:
                raise RuntimeError(
                    "megamoe_m2n remote shared expert currently supports exactly "
                    f"one GLM shared expert, got n_shared_experts={n_shared_experts}"
                )
            # Split MegaMoE M2N: one fused kernel per side per layer over the
            # union world; the adapter holds one symmetric buffer per lane.
            from minisgl.moe.megamoe_m2n_afd import MegaMoEM2NAfdAdapter

            mega_adapter = MegaMoEM2NAfdAdapter(
                group=dist.group.WORLD,
                ag_size=ag_size,
                eg_size=eg_size,
                real_num_experts=int(mc.num_experts),
                hidden_size=int(mc.hidden_size),
                intermediate_size=int(mc.moe_intermediate_size),
                top_k=int(mc.num_experts_per_tok),
                num_max_dispatch_tokens_per_rank=bucket,
                num_lanes=adapter_lanes,
                gate_renormalize=bool(getattr(mc, "norm_topk_prob", True)),
                shared_experts_per_rank=1 if remote_shared_experts else 0,
                routed_scaling_factor=float(model_extra.get("routed_scaling_factor", 1.0)),
            )
            # Lane view keeps the downstream `adapters[mb % len(adapters)]`
            # indexing working; every lane shares the adapter object and picks
            # its own symmetric buffer via the `lane` argument.
            self._afd_adapters = [mega_adapter] * adapter_lanes
            self._afd_adapter = mega_adapter
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_megamoe_m2n_adapter "
                f"lanes={adapter_lanes} afd_num_mb={int(self.afd_num_mb)} bucket={bucket} "
                f"remote_shared={int(remote_shared_experts)} "
                f"eg_expected_tokens_override={mega_adapter.eg_expected_tokens_override} "
                f"eg_expected_tokens_source={mega_adapter.eg_expected_tokens_source}",
                flush=True,
            )
        elif self._afd_moe_backend == "rccl":
            # ROCm/gfx942 path: DeepEP cannot build here (it needs the NCCL 2.28
            # device API, absent from RCCL, plus SM90 TMA/mbarrier). This transport
            # uses torch.distributed collectives instead -- correct but slower.
            # See dev_log/{10,11,12}*.md.
            from minisgl.moe.rccl_m2n_adapter import RcclM2NAdapter

            # The WORLD group is GLOO on this port: engine.py takes the gloo branch
            # whenever `tp_info.size == 1 or use_pynccl`, and use_pynccl is True on ROCm
            # because HIP graph capture needs it. Handing that group to the adapter put the
            # whole M2N payload -- bf16 activations, every layer, every forward -- on TCP
            # sockets with CUDA tensors staged through host memory. Measured at gpt-oss
            # shapes (dev_log/probes/_m2n_backend_worker.py, 2A+2F, H=2880, K=4): RCCL does
            # a dispatch+combine round trip in 0.843 ms/layer, while gloo does not complete
            # the SMALLEST case within 900 s. That is the HEAD serving hang, and doc 28
            # retracts the transport analysis built on the RCCL-only probe.
            #
            # So build a device-backed group spanning all AG+EG ranks for the payload. Every
            # rank reaches this branch, so new_group() is collective here as required.
            #
            # ONE GROUP PER LANE, not one shared by all of them. An RCCL communicator does not
            # support concurrent collectives issued from different streams: with afd_num_mb >= 2
            # the lanes have their own streams and both call `all_to_all_single` at once, and a
            # single shared communicator is then undefined behaviour.
            #
            # NECESSARY BUT NOT SUFFICIENT: this was tried as the fix for the num_mb=2 + graphs
            # crash (doc 42) and the crash PERSISTED, so the shared communicator was not the cause
            # either. Kept because concurrent collectives on one communicator are unsafe
            # regardless, and a shared group would corrupt as soon as the lanes do overlap.
            #
            # `new_group` is collective, and every rank reaches this branch and creates the same
            # count in the same order, so building `adapter_lanes` of them is safe here.
            m2n_groups = [dist.group.WORLD] * adapter_lanes
            m2n_backend = "gloo/WORLD(fallback)"
            try:
                dev_backend = "nccl" if torch.cuda.is_available() else None
                if dev_backend is not None:
                    m2n_groups = [
                        dist.new_group(backend=dev_backend) for _ in range(adapter_lanes)
                    ]
                    m2n_backend = dev_backend
            except Exception as exc:  # pragma: no cover - environment dependent
                # Never silently fall back to the path that hangs: a working-but-glacial
                # transport is far better than a wedged one, but an UNREPORTED downgrade is
                # what cost four documents of wrong attribution.
                m2n_groups = [dist.group.WORLD] * adapter_lanes
                m2n_backend = f"gloo/WORLD(fallback: {type(exc).__name__}: {exc})"

            self._afd_adapters = [
                RcclM2NAdapter(
                    group=m2n_groups[_lane],
                    ag_size=ag_size,
                    eg_size=eg_size,
                    real_num_experts=int(mc.num_experts),
                    hidden_size=int(mc.hidden_size),
                    top_k=int(mc.num_experts_per_tok),
                    num_max_dispatch_tokens_per_rank=bucket,
                    log=lambda m: log_line(self.log_path, m, flush=True),
                )
                for _lane in range(adapter_lanes)
            ]
            self._afd_adapter = self._afd_adapters[0]
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_rccl_m2n_adapters "
                f"lanes={adapter_lanes} afd_num_mb={int(self.afd_num_mb)} bucket={bucket} "
                # State the backend. A transport class named RcclM2NAdapter silently running
                # on gloo is exactly what hid this; one log line would have prevented it.
                f"m2n_backend={m2n_backend}",
                flush=True,
            )
        else:
            from minisgl.moe.deepep_m2n_adapter import DeepEPM2NAdapter

            if adapter_lanes > 1:
                os.environ["MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM"] = "1"
            self._afd_adapters = [
                DeepEPM2NAdapter(
                    group=dist.group.WORLD,
                    ag_size=ag_size,
                    eg_size=eg_size,
                    real_num_experts=int(mc.num_experts),
                    hidden_size=int(mc.hidden_size),
                    top_k=int(mc.num_experts_per_tok),
                    num_max_dispatch_tokens_per_rank=bucket,
                )
                for _lane in range(adapter_lanes)
            ]
            self._afd_adapter = self._afd_adapters[0]
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] afd_deepep_adapters "
                f"lanes={adapter_lanes} afd_num_mb={int(self.afd_num_mb)} bucket={bucket} "
                f"per_buffer_comm_stream={os.environ.get('MINISGL_DEEPEP_PER_BUFFER_COMM_STREAM', '0')}",
                flush=True,
            )

        # Keep fp8 weights and *_scale (float32) as-is. Router gates/biases
        # that need FP32 are listed below; everything else follows model dtype.
        is_minimax_m2 = arch0 == "MiniMaxM2ForCausalLM"
        is_glm4_moe = arch0 == "Glm4MoeForCausalLM"

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

        def _cast_full(pairs):
            tgt = config.dtype
            return {
                k: (
                    v
                    if not v.is_floating_point()
                    or v.dtype == torch.float8_e4m3fn
                    or (v.dtype == torch.float32 and k.endswith("_scale"))
                    else v.to(torch.float32)
                    if _minimax_fp32_key(k) or _glm4_fp32_key(k)
                    else v.to(tgt)
                )
                for k, v in pairs
            }

        def _afd_stage_classes():
            arch = (getattr(mc, "architectures", None) or [""])[0]
            if arch == "Qwen3MoeForCausalLM":
                from minisgl.models.qwen3_moe_afd import (
                    Qwen3AfdDenseRouterForCausalLM,
                    Qwen3AfdExpertStage,
                )

                return Qwen3AfdDenseRouterForCausalLM, Qwen3AfdExpertStage
            if arch == "MiniMaxM2ForCausalLM":
                from minisgl.models.minimax_m2_afd import (
                    MiniMaxM2AfdDenseRouterForCausalLM,
                    MiniMaxM2AfdExpertStage,
                )

                return MiniMaxM2AfdDenseRouterForCausalLM, MiniMaxM2AfdExpertStage
            if arch == "GptOssForCausalLM":
                from minisgl.models.gpt_oss_afd import (
                    GptOssAfdDenseRouterForCausalLM,
                    GptOssAfdExpertStage,
                )

                return GptOssAfdDenseRouterForCausalLM, GptOssAfdExpertStage
            if arch == "Glm4MoeForCausalLM":
                from minisgl.models.glm4_moe_afd import (
                    Glm4AfdDenseRouterForCausalLM,
                    Glm4AfdExpertStage,
                )

                return Glm4AfdDenseRouterForCausalLM, Glm4AfdExpertStage
            raise RuntimeError(f"AFD AG/EG does not support architecture {arch!r}")

        AfdDenseRouterForCausalLM, AfdExpertStage = _afd_stage_classes()

        if self.role == "attention":
            from minisgl.layers import set_rope_device

            set_rope_device(runtime_state.device)
            with torch.device("meta"), torch_dtype(config.dtype):
                model = AfdDenseRouterForCausalLM(mc)
            full = _cast_full(load_weight(config.model_path, runtime_state.device, skip_expert_weights=True))
            model.load_state_dict(extract_stage_state_dict(model, full))
            del full
            self._afd_ag_model = model
            self._afd_sampler = Sampler(runtime_state.device, int(mc.vocab_size))
        else:
            # The AFD EG stage receives already-dispatched DeepEPM2N payloads
            # from the outer adapter above. Its per-layer MoELayer objects are
            # only expert weight holders plus DeepGEMM runners; AFD EG model
            # classes explicitly disable MoELayer's internal A2A dispatcher.
            with torch.device("meta"), torch_dtype(config.dtype):
                model = AfdExpertStage(mc)
            full = _cast_full(
                load_weight(config.model_path, runtime_state.device, skip_non_expert_weights=True)
            )
            model.load_state_dict(extract_stage_state_dict(model, full))
            del full
            try:
                sd = model.state_dict()
                tensor_bytes = sum(int(t.numel() * t.element_size()) for t in sd.values())
                remote_layers = [
                    layer
                    for layer in getattr(model.model.layers, "op_list", [])
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")
                ]
                shapes = []
                for idx in (0, len(remote_layers) - 1):
                    if 0 <= idx < len(remote_layers):
                        experts = remote_layers[idx].mlp.experts
                        if getattr(experts, "_is_mxfp4_packed", False):
                            shape_text = (
                                f"gate_up_blocks={tuple(experts.gate_up_proj_blocks.shape)} "
                                f"down_blocks={tuple(experts.down_proj_blocks.shape)}"
                            )
                        else:
                            shape_text = (
                                f"gate_up={tuple(experts.gate_up_proj.shape)}:"
                                f"{experts.gate_up_proj.dtype} "
                                f"down={tuple(experts.down_proj.shape)}:"
                                f"{experts.down_proj.dtype}"
                            )
                        shapes.append(
                            "layer_index="
                            f"{idx} local_experts={int(getattr(experts, 'local_num_experts', -1))} "
                            f"{shape_text}"
                        )
                log_line(
                    self.log_path,
                    f"[{self.role} rank={self.tp_rank}] afd_eg_model_memory "
                    f"state_tensors={len(sd)} state_bytes={tensor_bytes} "
                    f"state_gib={tensor_bytes / (1024**3):.2f} "
                    f"cuda_alloc_gib={torch.cuda.memory_allocated() / (1024**3):.2f} "
                    f"cuda_reserved_gib={torch.cuda.memory_reserved() / (1024**3):.2f} "
                    + " ".join(shapes),
                    flush=True,
                )
            except Exception as exc:
                log_line(
                    self.log_path,
                    f"[{self.role} rank={self.tp_rank}] afd_eg_model_memory_log_failed "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if self._afd_moe_backend == "megamoe_m2n":
                # One-time per-layer requant of FP8 expert weights to the mega
                # kernel's per-32 UE8M0 interleaved format.
                layers = model.model.layers.op_list
                registered_layers = 0
                for layer_id, eg_layer in enumerate(layers):
                    experts = eg_layer.mlp.experts
                    shared_experts = getattr(eg_layer.mlp, "shared_experts", None)
                    self._afd_adapter.register_eg_layer_weights(
                        layer_id, experts, shared_experts=shared_experts
                    )
                    registered_layers += 1
                    # MegaMoE keeps its transformed weights in the adapter. Drop
                    # the Qwen-format tensors immediately; retaining both forms
                    # OOMs on the 235B expert ranks.
                    experts.gate_up_proj = None
                    experts.gate_up_proj_scale = None
                    experts.down_proj = None
                    experts.down_proj_scale = None
                    if shared_experts is not None:
                        shared_experts.gate_up_proj.weight = None
                        shared_experts.gate_up_proj.weight_scale = None
                        shared_experts.down_proj.weight = None
                        shared_experts.down_proj.weight_scale = None
                    if layer_id % 4 == 3:
                        torch.cuda.empty_cache()
                log_line(
                    self.log_path,
                    f"[{self.role} rank={self.tp_rank}] afd_megamoe_m2n_weights "
                    f"registered layers={registered_layers}/{len(layers)}",
                    flush=True,
                )
                # MegaMoE runtime owns the transformed weights in the adapter.
                # Drop the ExpertStage object too; the EG persistent and
                # per-layer MegaMoE paths never call run_experts().
                self._afd_eg_model = None
                del layers
                del model
                torch.cuda.empty_cache()
            else:
                self._afd_eg_model = model

        # CUDA-graph decode runners (lockstep AG/EG). Built here; captured later
        # by warmup_afd_decode_graphs() once the coordinator drives all ranks
        # together (the per-layer dispatch/combine are collectives).
        self._afd_ag_graph = None
        self._afd_eg_graph = None
        # Per-layer route graphs (doc 28 §14.2). Independent of enable_decode_graph and of
        # decode_graph_bs: this captures only `route`, which contains no M2N collective, so it
        # needs neither the AG/EG capture lockstep nor fixed-shape M2N. Off by default.
        self._ag_route_graphs = None
        self._ag_attn_graphs = None
        if self.role == "attention":
            from .afd_ag_compute_graph import (
                AfdAgAttnGraphs,
                AfdAgRouteGraphs,
                ag_attn_graph_enabled,
                ag_route_graph_enabled,
            )

            if ag_attn_graph_enabled():
                self._ag_attn_graphs = AfdAgAttnGraphs(
                    runtime_state=self.runtime_state,
                    model=self._afd_ag_model,
                    num_layers=int(self.num_layers),
                    hidden_size=int(self.hidden_size),
                    num_mb=int(self.afd_num_mb),
                    log_path=self.log_path,
                )

            if ag_route_graph_enabled():
                self._ag_route_graphs = AfdAgRouteGraphs(
                    runtime_state=self.runtime_state,
                    model=self._afd_ag_model,
                    num_layers=int(self.num_layers),
                    hidden_size=int(self.hidden_size),
                    num_mb=int(self.afd_num_mb),
                    log_path=self.log_path,
                )
                log_line(
                    self.log_path,
                    f"[{self.role} rank={self.tp_rank}] ag_route_graph ENABLED "
                    f"layers={self.num_layers}",
                    flush=True,
                )
        # Per-layer EG expert graphs (doc 38). Same rationale as the AG route/attn graphs and the
        # same independence: `run_experts` contains no M2N collective, so this needs neither the
        # AG/EG capture lockstep nor fixed-shape M2N. Off by default.
        self._eg_expert_graphs = None
        if self.role == "mlp":
            from .afd_eg_expert_graph import AfdEgExpertGraphs, eg_expert_graph_enabled

            adapter = getattr(self, "_afd_adapter", None)
            if eg_expert_graph_enabled() and adapter is not None:
                stage = getattr(self, "_afd_eg_expert_stage", None) or getattr(
                    self, "_afd_eg_model", None
                )
                if stage is not None:
                    self._eg_expert_graphs = AfdEgExpertGraphs(
                        runtime_state=self.runtime_state,
                        expert_stage=stage,
                        adapter=adapter,
                        num_layers=int(self.num_layers),
                        hidden_size=int(self.hidden_size),
                        num_mb=int(self.afd_num_mb),
                        log_path=self.log_path,
                    )
                    log_line(
                        self.log_path,
                        f"[{self.role} rank={self.tp_rank}] eg_expert_graph ENABLED "
                        f"layers={self.num_layers}",
                        flush=True,
                    )
                else:
                    log_line(
                        self.log_path,
                        f"[{self.role} rank={self.tp_rank}] eg_expert_graph requested but no "
                        "expert stage found; staying eager",
                        flush=True,
                    )
        if self.enable_decode_graph and self.decode_graph_bs:
            comm_stream = getattr(self, "_comm_stream", None)
            lane_comm_streams = tuple(getattr(self, "_afd_comm_streams", ()))
            if self.role == "attention":
                from .afd_attention_worker import AfdAgDecodeGraphRunner

                self._afd_ag_graph = AfdAgDecodeGraphRunner(
                    runtime_state=runtime_state,
                    model=self._afd_ag_model,
                    adapters=self._afd_adapters,
                    num_mb=self.afd_num_mb,
                    comm_stream=comm_stream,
                    lane_comm_streams=lane_comm_streams,
                    runtime=self.runtime,
                    num_layers=int(mc.num_layers),
                    hidden_size=int(mc.hidden_size),
                    bs_list=tuple(self.decode_graph_bs),
                    log_path=self.log_path,
                    label=f"{self.role} rank={self.tp_rank}",
                )
            else:
                from .afd_expert_worker import AfdEgDecodeGraphRunner

                self._afd_eg_graph = AfdEgDecodeGraphRunner(
                    runtime_state=runtime_state,
                    expert_stage=self._afd_eg_model,
                    adapters=self._afd_adapters,
                    num_mb=self.afd_num_mb,
                    comm_stream=comm_stream,
                    lane_comm_streams=lane_comm_streams,
                    num_layers=int(self.num_layers),
                    hidden_size=int(mc.hidden_size),
                    bs_list=tuple(self.decode_graph_bs),
                    log_path=self.log_path,
                    label=f"{self.role} rank={self.tp_rank}",
                )

        self._afd_initialized = True
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] afd_runtime:init_done "
            f"ag={ag_size} eg={eg_size} bucket={bucket} "
            f"raw_max_comm_tokens={int(getattr(self, 'max_comm_tokens', 0))} "
            f"dp_fanin={dp_fanin}",
            flush=True,
        )

    def warmup_afd_decode_graphs(self, bs_list) -> dict[str, Any]:
        """Capture the AG/EG decode graphs for each bucket. MUST be invoked on all
        AG+EG ranks concurrently (the coordinator ray.gets them together) so the
        per-bucket capture — which runs the DeepEP dispatch/combine COLLECTIVES —
        stays in lockstep across ranks. Buckets captured ascending, same order on
        every rank."""
        if not self.enable_decode_graph:
            return {"role": self.role, "rank": self.tp_rank, "captured": []}
        runner = self._afd_ag_graph if self.role == "attention" else self._afd_eg_graph
        if runner is None:
            return {"role": self.role, "rank": self.tp_rank, "captured": []}
        captured = []
        for bs in sorted(int(b) for b in bs_list):
            runner.warmup(bs)
            captured.append(bs)
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] afd_decode_graphs:captured {captured}",
            flush=True,
        )
        return {"role": self.role, "rank": self.tp_rank, "captured": captured}

    def _detach_nsys_injection_from_subprocesses(self) -> None:
        """Keep profiler injection in the worker and sanitize only build children.

        DeepEP/DeepGEMM builds pass sanitized envs through build_utils. Mutating
        the worker process env here breaks NVTX collection for all later ranges.
        """
        active = [
            key
            for key in (
            "CUDA_INJECTION64_PATH",
            "CUDA_INJECTION32_PATH",
            "NVTX_INJECTION64_PATH",
            "NVTX_INJECTION32_PATH",
            )
            if key in os.environ
        ]
        if active or os.environ.get("LD_PRELOAD"):
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] nsys_injection:preserved "
                f"keys={active} ld_preload_len={len(os.environ.get('LD_PRELOAD', ''))}",
                flush=True,
            )

    def start_cuda_profiler(self, sync: bool = True) -> dict[str, Any]:
        if not self._ray_nsys_enabled or self._nsys_runtime_started:
            return {"role": self.role, "rank": self.tp_rank, "started": self._nsys_runtime_started}
        if sync:
            torch.cuda.synchronize(self.device)
        torch.cuda.profiler.start()
        self._nsys_runtime_started = True
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] nsys profiler:start sync={int(bool(sync))}",
            flush=True,
        )
        return {"role": self.role, "rank": self.tp_rank, "started": True}

    def stop_cuda_profiler(self, sync: bool = True) -> dict[str, Any]:
        if not self._ray_nsys_enabled or not self._nsys_runtime_started:
            return {"role": self.role, "rank": self.tp_rank, "stopped": False}
        if sync:
            torch.cuda.synchronize(self.device)
        torch.cuda.profiler.stop()
        self._nsys_runtime_started = False
        log_line(
            self.log_path,
            f"[{self.role} rank={self.tp_rank}] nsys profiler:stop sync={int(bool(sync))}",
            flush=True,
        )
        return {"role": self.role, "rank": self.tp_rank, "stopped": True}

    def handle_profiler_cmd(self, cmd: AfdProfilerCmd) -> None:
        if cmd.action == "start":
            self.start_cuda_profiler(sync=bool(cmd.sync))
            return
        if cmd.action == "stop":
            self.stop_cuda_profiler(sync=bool(cmd.sync))
            return
        raise RuntimeError(f"Unsupported AFD profiler command action: {cmd.action!r}")

    def _send_reply(self, reply: AfdReply) -> None:
        queue = self._send_to_coordinator
        if queue is None:
            raise RuntimeError("Hot RPC loop is not initialized")
        step_id = None if reply.step_id is None else int(reply.step_id)
        fields = {
            "step": step_id,
            "kind": type(reply).__name__,
            "worker_rank": self.worker_rank,
            "tp_rank": self.tp_rank,
        }
        prefix = "AFD_Worker_SendReply"
        serialize_start_ns = time.perf_counter_ns()
        with nvtx_range(nvtx_label(f"{prefix}_Serialize", **fields)):
            raw = queue.serialize(reply)
        serialize_done_ns = time.perf_counter_ns()
        with nvtx_range(nvtx_label(f"{prefix}_Put", bytes=len(raw), **fields)):
            queue.put_raw(raw)
        put_done_ns = time.perf_counter_ns()
        if step_id is not None and self._control_profile_enabled(step_id):
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] reply_send kind={type(reply).__name__} "
                f"step_id={step_id} worker_rank={self.worker_rank} bytes={len(raw)} "
                f"serialize_ms={(serialize_done_ns - serialize_start_ns) / 1_000_000.0:.3f} "
                f"put_ms={(put_done_ns - serialize_done_ns) / 1_000_000.0:.3f} "
                f"total_ms={(put_done_ns - serialize_start_ns) / 1_000_000.0:.3f}",
            )

    def _warmup_fused_plan(self) -> None:
        """Pre-compile the fused send-plan kernel so no request pays Triton JIT.

        Left lazy it cost 25.2 s on the first 48-token completion against 1.55 s once warm, and
        steady-state profiling could not see it (the sub-step capture skips the first 8 steps and
        reported a healthy 42.79 ms). Attention role only: EG ranks source no tokens and never
        build a send plan.
        """
        adapter = getattr(self, "_afd_adapter", None)
        if adapter is None:
            return
        try:
            from minisgl.moe import m2n_plan_triton as fused
            from minisgl.moe.rccl_m2n_adapter import _FUSED_GROUP, _FUSED_PLAN

            dev = torch.device("cuda")
            n_plan = n_group = 0
            # The send plan runs on AG, the grouping on EG -- warm each where it will actually run.
            if self.role == "attention" and _FUSED_PLAN:
                n_plan = fused.warmup(
                    dev,
                    ag_size=adapter.layout.ag_size,
                    experts_per_eg_rank=adapter.experts_per_eg_rank,
                    world=adapter.world_size,
                    top_k=adapter.top_k,
                )
            if self.role == "mlp" and _FUSED_GROUP:
                local_experts = adapter.experts_per_eg_rank
                n_group = fused.warmup_group(
                    dev,
                    first_local=(adapter.rank - adapter.layout.ag_size) * local_experts,
                    local_experts=local_experts,
                )
            if not (n_plan or n_group):
                return
            log_line(self.log_path,
                     f"[{self.role} rank={self.tp_rank}] fused_plan:warmed "
                     f"plan={n_plan} group={n_group}",
                     flush=True)
        except Exception as exc:
            # Never fatal: the adapter falls back to the eager build, which is correct and only
            # slower. A warmup failure must not take the server down.
            log_line(self.log_path,
                     f"[{self.role} rank={self.tp_rank}] fused_plan:warmup_failed "
                     f"{type(exc).__name__}: {exc}",
                     flush=True)

    def _send_hot_loop_ready(self) -> None:
        self._send_reply(
            AfdWorkerReadyReply(
                worker_rank=self.worker_rank,
                step_id=None,
            )
        )

    def start_hot_rpc_loop(
        self,
        cmd_addr: str,
        result_addr: str,
        detokenizer_addr: str | None = None,
        create_detokenizer_link: bool = False,
    ) -> dict[str, Any]:
        self._recv_from_coordinator = ZmqPullQueue(
            cmd_addr,
            create=False,
            decoder=AfdCommand.decoder,
            serde="pickle",
        )
        self._send_to_coordinator = ZmqPushQueue(
            result_addr,
            create=False,
            encoder=AfdReply.encoder,
            serde="pickle",
        )
        if detokenizer_addr is not None and self.role == "attention" and self.tp_rank == 0:
            self._send_into_detokenizer = ZmqPushQueue(
                detokenizer_addr,
                create=bool(create_detokenizer_link),
                encoder=BaseTokenizerMsg.encoder,
            )
        self._hot_loop_running = True
        log_line(self.log_path, f"[{self.role} rank={self.tp_rank}] hot_rpc_loop:start", flush=True)
        try:
            self._run_hot_rpc_loop()
        except Exception:
            log_line(
                self.log_path,
                f"[{self.role} rank={self.tp_rank}] hot_rpc_loop:error\n{traceback.format_exc()}",
                flush=True,
            )
            raise
        finally:
            self._hot_loop_running = False
            # Finalize profiling here too: if the coordinator is killed before it
            # can call worker.shutdown() (observed with attn_dp>1 multi-node
            # teardown), this is the only place the per-rank cpu_trace + nsys-rep
            # get flushed.
            self._finalize_profiling()
            for attr in (
                "_recv_from_coordinator",
                "_send_to_coordinator",
                "_send_into_detokenizer",
            ):
                self._safe_close(attr)
            log_line(self.log_path, f"[{self.role} rank={self.tp_rank}] hot_rpc_loop:stop", flush=True)
        return {"role": self.role, "rank": self.tp_rank, "stopped": True}

    def _finalize_profiling(self) -> None:
        # Flush the CPU NVTX trace + finalize the nsys capture. Idempotent so it
        # can run from either the hot-loop finally or shutdown(), whichever fires
        # first. Needed because the coordinator's graceful shutdown (which calls
        # worker.shutdown()) can be killed before it runs under attn_dp>1 /
        # multi-node teardown, leaving the per-rank cpu_trace + nsys-rep unwritten.
        if getattr(self, "_profiling_finalized", False):
            return
        self._profiling_finalized = True
        if self._cpu_trace_path is not None:
            log_line(self.log_path, f"[{self.role} rank={self.tp_rank}] cpu_trace flush:start", flush=True)
        try:
            trace_path = shutdown_nvtx_cpu_trace()
            if trace_path:
                log_line(
                    self.log_path,
                    f"[{self.role} rank={self.tp_rank}] cpu_trace flushed path={trace_path}",
                    flush=True,
                )
        except Exception:
            pass
        try:
            self.stop_cuda_profiler()
        except Exception:
            pass

    def shutdown(self) -> None:
        self._finalize_profiling()
        for attr in (
            "_recv_from_coordinator",
            "_send_to_coordinator",
        ):
            self._safe_close(attr)
        self._safe_close("runtime", method="shutdown")
        self._destroy_afd_adapters()
        flush_log_lines(self.log_path)

    def _destroy_afd_adapters(self) -> None:
        adapters = list(getattr(self, "_afd_adapters", []))
        self._afd_adapters = []
        self._afd_adapter = None
        for adapter in reversed(adapters):
            try:
                adapter.destroy()
            except Exception:
                pass

    def _safe_close(self, attr: str, *, method: str = "stop") -> None:
        obj = getattr(self, attr, None)
        if obj is None:
            return
        try:
            getattr(obj, method)()
        except Exception:
            pass
        setattr(self, attr, None)
