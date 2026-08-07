from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from minisgl.core import Batch, Req
from minisgl.kvcache import MatchResult
from minisgl.kvcache.naive_cache import NaiveCacheHandle
from minisgl.message import DetokenizeMsg, UserMsg
from minisgl.scheduler.decode import DecodeManager
from minisgl.scheduler.prefill import PrefillManager
from minisgl.scheduler.scheduler import _complete_batch_results
from minisgl.scheduler.table import TableManager
from minisgl.utils import div_ceil

from .afd_protocol import AfdModelPlan
from .afd_support import select_decode_graph_bs


@dataclass(frozen=True)
class AfdBatchCompletion:
    replies: list[DetokenizeMsg]
    freed_table_indices: tuple[int, ...]
    reply_uids: list[int] | None = None
    reply_next_tokens: list[int] | None = None
    reply_finished: list[bool] | None = None


class _CpuNaiveCacheManager:
    """Coordinator-side KV page ledger for centralized AFD scheduling.

    This intentionally implements only the small CacheManager surface used by
    PrefillManager/_complete_batch_results, and only for cache_type='naive'.
    GPU page-table writes are returned as compact deltas for attention workers.
    """

    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor):
        self.num_pages = int(num_pages)
        self.page_size = int(page_size)
        self.page_table = page_table
        self.free_slots = list(range(0, self.num_pages * self.page_size, self.page_size))
        self._page_offsets = (
            torch.arange(self.page_size, dtype=torch.int32)
            if self.page_size != 1
            else None
        )

    def match_req(self, req) -> MatchResult:
        if req.input_len <= 0:
            raise RuntimeError("AFD centralized scheduler got empty input request")
        return MatchResult(NaiveCacheHandle())

    @property
    def available_size(self) -> int:
        return len(self.free_slots) * self.page_size

    def lock(self, handle) -> None:
        del handle

    def unlock(self, handle) -> None:
        del handle

    def allocate_paged(self, reqs: list[Req]) -> None:
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)
            last_page = div_ceil(req.device_len, self.page_size)
            needed_pages = last_page - first_page
            if needed_pages <= 0:
                continue
            if needed_pages > len(self.free_slots):
                raise RuntimeError(
                    "AFD centralized naive cache exhausted: "
                    f"needed_pages={needed_pages} free_pages={len(self.free_slots)}"
                )
            pages = self.free_slots[:needed_pages]
            del self.free_slots[:needed_pages]
            values = self._page_to_token_tensor(pages)
            start = first_page * self.page_size
            end = last_page * self.page_size
            self.page_table[int(req.table_idx), start:end] = values

    def cache_req(self, req: Req, *, finished: bool) -> None:
        if not finished:
            return
        if int(req.cached_len) <= 0:
            return
        page_tokens = self.page_table[int(req.table_idx), : int(req.cached_len)]
        self._free(page_tokens)

    def check_integrity(self) -> None:
        if len(self.free_slots) > self.num_pages:
            raise RuntimeError("AFD centralized naive cache has too many free pages")

    @contextmanager
    def lazy_free_region(self):
        yield

    def _free(self, token_indices: torch.Tensor) -> None:
        if int(token_indices.numel()) == 0:
            return
        self.free_slots.extend(int(x) for x in token_indices[:: self.page_size].tolist())

    def _page_to_token_tensor(self, pages: list[int]) -> torch.Tensor:
        page_tensor = torch.tensor(pages, dtype=torch.int32)
        if self.page_size == 1:
            return page_tensor
        assert self._page_offsets is not None
        return (page_tensor.unsqueeze(1) + self._page_offsets).flatten()


def _compute_microbatch_offsets(
    batch: Batch,
    num_mb: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split one local DP batch into request and flat-token microbatch offsets."""
    num_mb = max(1, int(num_mb))
    padded_reqs = batch.padded_reqs
    size = len(padded_reqs)
    if batch.phase == "decode":
        if num_mb == 1 or size < num_mb:
            return (0, size), (0, size)
        base = size // num_mb
        rem = size % num_mb
        offsets = [0]
        for mb in range(num_mb):
            offsets.append(offsets[-1] + base + (1 if mb < rem else 0))
        offsets_t = tuple(offsets)
        return offsets_t, offsets_t

    extend_lens = [int(req.extend_len) for req in padded_reqs]
    total_tokens = sum(extend_lens)
    if num_mb == 1 or size < num_mb:
        return (0, size), (0, total_tokens)

    cumsum = [0]
    for length in extend_lens:
        cumsum.append(cumsum[-1] + length)
    req_offsets = [0]
    token_offsets = [0]
    for mb in range(1, num_mb):
        target = total_tokens * mb // num_mb
        lo = req_offsets[-1] + 1
        hi = size - (num_mb - mb)
        best = lo
        best_delta = abs(cumsum[lo] - target)
        for idx in range(lo + 1, hi + 1):
            delta = abs(cumsum[idx] - target)
            if delta >= best_delta:
                break
            best = idx
            best_delta = delta
        req_offsets.append(best)
        token_offsets.append(cumsum[best])
    req_offsets.append(size)
    token_offsets.append(total_tokens)
    return tuple(req_offsets), tuple(token_offsets)


class CentralizedAfdDpScheduler:
    """One coordinator-owned CPU scheduler state for one attention DP."""

    def __init__(
        self,
        *,
        dp_rank: int,
        max_running_req: int,
        max_seq_len: int,
        num_pages: int,
        page_size: int,
        prefill_budget: int,
        decode_graph_bs: tuple[int, ...],
        cuda_graph_max_bs: int,
        afd_num_mb: int,
        mlp_fanout: int,
        eos_token_id: int,
    ):
        self.max_running_req = int(max_running_req)
        self.max_seq_len = int(max_seq_len)
        self.prefill_budget = int(prefill_budget)
        self.decode_graph_bs = tuple(int(x) for x in decode_graph_bs)
        self.decode_graph_bucket_set = set(self.decode_graph_bs)
        self.cuda_graph_max_bs = int(cuda_graph_max_bs)
        self.afd_num_mb = max(1, int(afd_num_mb))
        self.mlp_fanout = max(1, int(mlp_fanout))
        self.eos_token_id = int(eos_token_id)
        aligned_max_seq_len = ((self.max_seq_len + 31) // 32) * 32
        self.page_table = torch.zeros(
            (self.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
        )
        self.page_table[self.max_running_req].fill_(int(num_pages) * int(page_size))
        self.table_manager = TableManager(
            self.max_running_req,
            self.page_table,
            with_token_pool=False,
        )
        self.cache_manager = _CpuNaiveCacheManager(
            int(num_pages),
            int(page_size),
            self.page_table,
        )
        self.decode_manager = DecodeManager(int(page_size))
        self.prefill_manager = PrefillManager(
            self.cache_manager,  # type: ignore[arg-type]
            self.table_manager,
            self.decode_manager,
        )
        self.finished_reqs: set[Req] = set()
        self._dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32),
            table_idx=self.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=NaiveCacheHandle(),
        )

    def add_reqs(self, msgs: list[UserMsg]) -> None:
        self.prefill_manager.add_reqs(msgs)

    def abort_req(self, uid: int) -> tuple[int, ...]:
        req_to_free = self.prefill_manager.abort_req(int(uid))
        req_to_free = req_to_free or self.decode_manager.abort_req(int(uid))
        if req_to_free is None:
            return ()
        table_idx = int(req_to_free.table_idx)
        self._free_req_resources(req_to_free)
        return (table_idx,)

    def has_runnable_reqs(self) -> bool:
        return bool(self.prefill_manager.runnable or self.decode_manager.runnable)

    def schedule_step(
        self,
        step_id: int,
        *,
        phase: str,
    ) -> tuple[AfdModelPlan, Batch] | None:
        if phase == "prefill":
            batch = self.prefill_manager.schedule_next_batch(self.prefill_budget)
        elif phase == "decode":
            batch = self.decode_manager.schedule_next_batch()
        else:
            raise RuntimeError(f"Unsupported AFD schedule phase: {phase}")
        if batch is None:
            return None
        graph_token_count = self._pad_and_split_batch(batch)
        self.cache_manager.allocate_paged(batch.reqs)
        plan = AfdModelPlan.from_batch(
            int(step_id),
            batch,
            graph_token_count=graph_token_count,
        )
        return plan, batch

    def dummy_decode_step(
        self,
        step_id: int,
        *,
        padded_size: int,
    ) -> tuple[AfdModelPlan, Batch]:
        """Build a no-result AG participant so collective steps include every DP.

        The batch has zero real requests, but padded dummy decode tokens make the
        AG worker run the same MoE collective path and return an empty token list.
        """
        padded_size = max(self.afd_num_mb, int(padded_size))
        batch = Batch(reqs=[], phase="decode")
        batch.padded_reqs = [self._dummy_req] * padded_size
        req_offsets, tok_offsets = _compute_microbatch_offsets(
            batch,
            self.afd_num_mb,
        )
        batch.afd_microbatch_offsets = req_offsets
        batch.afd_microbatch_token_offsets = tok_offsets
        graph_token_count = int(len(batch.padded_reqs))
        plan = AfdModelPlan.from_batch(
            int(step_id),
            batch,
            graph_token_count=graph_token_count,
        )
        return plan, batch

    def complete_results(self, batch: Batch, next_tokens: list[int]) -> AfdBatchCompletion:
        if batch.is_decode:
            return self._complete_decode_results_fast(batch, next_tokens)
        replies, new_finished = _complete_batch_results(
            batch=batch,
            next_tokens_cpu=torch.tensor(next_tokens, dtype=torch.int32),
            eos_token_id=self.eos_token_id,
            finished_reqs=self.finished_reqs,
            decode_manager=self.decode_manager,
            cache_manager=self.cache_manager,  # type: ignore[arg-type]
            free_req_resources=self._free_req_resources,
        )
        self.finished_reqs = new_finished
        return AfdBatchCompletion(
            replies=replies,
            freed_table_indices=tuple(int(req.table_idx) for req in new_finished),
        )

    def _complete_decode_results_fast(
        self,
        batch: Batch,
        next_tokens: list[int],
    ) -> AfdBatchCompletion:
        """Complete an AFD decode step without per-token CPU tensor append.

        AFD advances `cached_len/device_len` before launching the AG command.
        The decode completion path only needs to emit the sampled token and
        remove finished requests. Keeping a full CPU `req.input_ids` by
        torch.cat-ing one token per request is O(context_len) work per token and
        dominates multi-node coordinator time.
        """
        reply_uids: list[int] = []
        reply_next_tokens: list[int] = []
        reply_finished: list[bool] = []
        new_finished: set[Req] = set()
        eos_token_id = self.eos_token_id
        eos_set = (
            {int(v) for v in eos_token_id}
            if isinstance(eos_token_id, (list, tuple, set))
            else None
        )
        finished_reqs = self.finished_reqs
        decode_manager = self.decode_manager
        free_req_resources = self._free_req_resources
        for i, req in enumerate(batch.reqs):
            next_token_id = int(next_tokens[i]) if i < len(next_tokens) else 0
            completed_len = int(getattr(req, "afd_completed_device_len", len(req.input_ids))) + 1
            req.afd_completed_device_len = completed_len
            finished = completed_len >= int(req.max_device_len)
            if not req.sampling_params.ignore_eos:
                if eos_set is not None:
                    finished |= next_token_id in eos_set
                elif eos_token_id is not None:
                    finished |= next_token_id == int(eos_token_id)
            reply_uids.append(int(req.uid))
            reply_next_tokens.append(next_token_id)
            reply_finished.append(bool(finished))
            if finished and req not in finished_reqs:
                decode_manager.remove_req(req)
                free_req_resources(req)
                new_finished.add(req)
        self.finished_reqs = new_finished
        return AfdBatchCompletion(
            replies=[],
            freed_table_indices=tuple(int(req.table_idx) for req in new_finished),
            reply_uids=reply_uids,
            reply_next_tokens=reply_next_tokens,
            reply_finished=reply_finished,
        )

    def _pad_and_split_batch(self, batch: Batch) -> int:
        m = self.afd_num_mb
        batch.afd_microbatch_offsets = None
        batch.afd_microbatch_token_offsets = None
        padded_size = int(
            self._select_padded_size(
                phase=str(batch.phase),
                real_size=int(batch.size),
            )
        )
        if padded_size < batch.size:
            raise RuntimeError(
                f"AFD padded_size smaller than batch size: "
                f"padded={padded_size} size={batch.size}"
            )
        balanced_reqs = self._balanced_decode_padded_reqs(batch, padded_size)
        if balanced_reqs is not None:
            batch.padded_reqs = balanced_reqs
        else:
            batch.padded_reqs = (
                batch.reqs + [self._dummy_req] * (padded_size - batch.size)
                if padded_size > batch.size
                else list(batch.reqs)
            )
        req_offsets, tok_offsets = _compute_microbatch_offsets(batch, m)
        batch.afd_microbatch_offsets = req_offsets
        batch.afd_microbatch_token_offsets = tok_offsets
        return int(sum(req.extend_len for req in batch.padded_reqs))

    def _balanced_decode_padded_reqs(
        self,
        batch: Batch,
        padded_size: int,
    ) -> list[Req] | None:
        """Distribute decode real requests across MB graph slots.

        AFD uses multi-MB graph replay. If decode padding is appended only at
        the end, a case like real=64, padded=128, mb=2 becomes mb0=64 real and
        mb1=64 dummy, which destroys overlap analysis and runs one real MB plus
        one empty MB. Keep each MB slice as a real prefix followed by dummy tail.
        """

        if str(batch.phase) != "decode":
            return None
        m = max(1, int(self.afd_num_mb))
        real_size = int(batch.size)
        padded_size = int(padded_size)
        if m <= 1 or padded_size <= real_size:
            return None
        if any(int(req.extend_len) != 1 for req in batch.reqs):
            return None
        base = padded_size // m
        rem = padded_size % m
        slot_sizes = [base + (1 if mb < rem else 0) for mb in range(m)]
        real_base = real_size // m
        real_rem = real_size % m
        real_counts = [real_base + (1 if mb < real_rem else 0) for mb in range(m)]
        if any(real_counts[mb] > slot_sizes[mb] for mb in range(m)):
            return None
        cursor = 0
        padded_reqs: list[Req] = []
        for mb in range(m):
            real_count = int(real_counts[mb])
            slot_size = int(slot_sizes[mb])
            padded_reqs.extend(batch.reqs[cursor : cursor + real_count])
            cursor += real_count
            padded_reqs.extend([self._dummy_req] * (slot_size - real_count))
        return padded_reqs

    def _select_padded_size(self, *, phase: str, real_size: int) -> int:
        if real_size <= 0:
            return 0
        m = self.afd_num_mb
        if phase != "decode" and m > 1:
            return max(int(real_size), int(m))
        if self.mlp_fanout > 1:
            if self.decode_graph_bs and phase == "decode":
                per_mb = (int(real_size) + m - 1) // m
                # Keep attention slots sized by their own per-MB work; MLP graph
                # padding is applied later when materializing model work items.
                try:
                    padded_per_mb = select_decode_graph_bs(per_mb, self.decode_graph_bs)
                    if padded_per_mb in self.decode_graph_bucket_set:
                        return int(padded_per_mb) * m
                except RuntimeError:
                    pass
                padded_per_mb = next(
                    (
                        bucket
                        for bucket in self.decode_graph_bs
                        if bucket >= per_mb
                        and bucket % self.mlp_fanout == 0
                        and (bucket // self.mlp_fanout) in self.decode_graph_bucket_set
                    ),
                    per_mb,
                )
                return int(padded_per_mb) * m
            return int(real_size)
        per_mb = (int(real_size) + m - 1) // m
        if self.decode_graph_bs and phase == "decode" and per_mb <= self.cuda_graph_max_bs:
            return int(select_decode_graph_bs(per_mb, self.decode_graph_bs)) * m
        return int(real_size)

    def _free_req_resources(self, req: Req) -> None:
        self.table_manager.free(int(req.table_idx))
        self.cache_manager.cache_req(req, finished=True)


__all__ = ["CentralizedAfdDpScheduler"]
