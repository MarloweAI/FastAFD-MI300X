from __future__ import annotations

import atexit
import json
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any


_STEP_RE = re.compile(r"\[step=(?P<step_id>\d+)\]")
_FIELD_RE = re.compile(r"\[(?P<key>[A-Za-z0-9_]+)=(?P<value>[^\]]+)\]")
_TRUTHY = {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def parse_nvtx_label_fields(name: str) -> dict[str, str]:
    return {match.group("key"): match.group("value") for match in _FIELD_RE.finditer(name)}


def parse_step_id_from_nvtx_name(name: str) -> int | None:
    match = _STEP_RE.search(name)
    if match is None:
        return None
    return int(match.group("step_id"))


class _CpuTraceRecorder:
    def __init__(
        self,
        output_path: Path,
        *,
        process_label: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.output_path = output_path
        self.process_label = process_label
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("process_label", process_label)
        self.metadata.setdefault("hostname", socket.gethostname())
        self.metadata.setdefault("pid", os.getpid())
        self.metadata.setdefault("created_time_ns", time.time_ns())
        self.metadata.setdefault("created_perf_ns", time.perf_counter_ns())
        self._lock = threading.Lock()
        self._local = threading.local()
        self._events: list[dict[str, Any]] = []
        self._next_id = 1

    def _stack(self) -> list[dict[str, Any]]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _alloc_id(self) -> int:
        with self._lock:
            event_id = self._next_id
            self._next_id += 1
            return event_id

    def enter(self, name: str) -> dict[str, Any]:
        stack = self._stack()
        fields = parse_nvtx_label_fields(name)
        explicit_step_id = parse_step_id_from_nvtx_name(name)
        inherited_step_id = stack[-1]["step_id"] if stack else None
        step_id = explicit_step_id if explicit_step_id is not None else inherited_step_id
        record = {
            "id": self._alloc_id(),
            "parent_id": stack[-1]["id"] if stack else None,
            "depth": len(stack),
            "name": name,
            "fields": fields,
            "step_id": step_id,
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name,
            "start_perf_ns": time.perf_counter_ns(),
            "start_time_ns": time.time_ns(),
        }
        stack.append(record)
        return record

    def exit(self, record: dict[str, Any], exc: BaseException | None = None) -> None:
        stack = self._stack()
        for index in range(len(stack) - 1, -1, -1):
            if stack[index] is record:
                stack.pop(index)
                break
        record["end_perf_ns"] = time.perf_counter_ns()
        record["end_time_ns"] = time.time_ns()
        record["duration_ns"] = int(record["end_perf_ns"]) - int(record["start_perf_ns"])
        if exc is not None:
            record["error"] = repr(exc)
        with self._lock:
            self._events.append(record.copy())

    def flush(self) -> str:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "metadata": self.metadata,
                "events": sorted(
                    self._events,
                    key=lambda item: (
                        int(item["start_time_ns"]),
                        int(item["thread_id"]),
                        int(item["depth"]),
                        int(item["id"]),
                    ),
                ),
            }
        tmp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, separators=(",", ":"))
        os.replace(tmp_path, self.output_path)
        return str(self.output_path)


_CPU_TRACE_RECORDER: _CpuTraceRecorder | None = None
_CPU_TRACE_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _resolve_enabled(explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return (
        os.environ.get("MINISGL_NVTX_CPU_TRACE", "") in _TRUTHY
        or os.environ.get("NSYS", "") in _TRUTHY
    )


def init_nvtx_cpu_trace(
    *,
    output_dir: str | Path,
    process_label: str,
    metadata: dict[str, Any] | None = None,
    enabled: bool | None = None,
) -> str | None:
    global _CPU_TRACE_RECORDER, _ATEXIT_REGISTERED
    if not _resolve_enabled(enabled):
        return None
    with _CPU_TRACE_LOCK:
        if _CPU_TRACE_RECORDER is not None:
            return str(_CPU_TRACE_RECORDER.output_path)
        output_path = Path(output_dir) / f"{process_label}_nvtx_cpu.json"
        _CPU_TRACE_RECORDER = _CpuTraceRecorder(
            output_path=output_path,
            process_label=process_label,
            metadata=metadata,
        )
        if not _ATEXIT_REGISTERED:
            atexit.register(flush_nvtx_cpu_trace)
            _ATEXIT_REGISTERED = True
        return str(output_path)


def enter_nvtx_cpu_trace(name: str) -> dict[str, Any] | None:
    recorder = _CPU_TRACE_RECORDER
    if recorder is None:
        return None
    return recorder.enter(name)


def exit_nvtx_cpu_trace(record: dict[str, Any] | None, exc: BaseException | None = None) -> None:
    recorder = _CPU_TRACE_RECORDER
    if recorder is None or record is None:
        return
    recorder.exit(record, exc)


def flush_nvtx_cpu_trace() -> str | None:
    recorder = _CPU_TRACE_RECORDER
    if recorder is None:
        return None
    return recorder.flush()


def shutdown_nvtx_cpu_trace() -> str | None:
    return flush_nvtx_cpu_trace()
