#!/usr/bin/env python3
"""Readable reports for FastAFD rocprof measurement CSV files.

Examples:
  profile_report.py results/profiles/RUN
  profile_report.py results/profiles/RUN --view summary --sort average --rank 0
  profile_report.py results/profiles/RUN --view timeline --rank 0,1 --limit 100
  profile_report.py results/profiles/RUN --view pattern --rank all

"Rank" is a convenient zero-based label assigned by sorting the GPU Agent IDs
present in the trace. Use --agent when the exact rocprof Agent ID matters.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import pathlib
import re
import statistics
import sys
from collections import defaultdict


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile",
        help="profile directory or measurement_kernel_times.csv",
    )
    parser.add_argument(
        "--view",
        choices=("summary", "timeline", "pattern"),
        default="summary",
        help="aggregate, chronological, or repeated-layer report (default: summary)",
    )
    parser.add_argument(
        "--rank",
        default="all",
        help="logical ranks to show: all, 0, or comma-separated 0,1",
    )
    parser.add_argument(
        "--agent",
        default="",
        help="exact rocprof agents to show, e.g. 2 or 2,3; overrides --rank",
    )
    parser.add_argument(
        "--sort",
        choices=("total", "average", "max", "calls", "name"),
        default="total",
        help="summary ordering (default: total)",
    )
    parser.add_argument(
        "--combine-ranks",
        action="store_true",
        help="combine the same kernel across selected ranks in summary view",
    )
    parser.add_argument("--kernel", default="", help="kernel-name regular expression")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="rows to print; 0 means all (default: 40, or all kernels with --step)",
    )
    parser.add_argument("--offset", type=int, default=0, help="rows to skip")
    parser.add_argument("--name-width", type=int, default=75)
    parser.add_argument("--full-names", action="store_true")
    parser.add_argument(
        "--model-config",
        help="config.json used to classify sliding/full-attention layers",
    )
    parser.add_argument("--layers", type=int, help="override transformer layer count")
    parser.add_argument(
        "--layer-marker",
        default=r"_paged_decode_split_kernel",
        help="kernel regex marking one decode attention layer",
    )
    parser.add_argument(
        "--step",
        type=int,
        help="in pattern view, print every kernel in one zero-based decode iteration",
    )
    parser.add_argument(
        "--step-marker",
        default=r"masked_index_kernel",
        help="kernel regex marking the beginning of a complete decode iteration",
    )
    parser.add_argument(
        "--markers-only",
        action="store_true",
        help="with --step, show only one attention marker per layer (old compact view)",
    )
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 0) or args.offset < 0 or args.name_width < 20:
        parser.error("--limit/--offset must be non-negative and --name-width must be >= 20")
    return args


def resolve_profile(value: str) -> tuple[pathlib.Path, pathlib.Path]:
    supplied = pathlib.Path(value).expanduser()
    if supplied.is_dir():
        profile_dir = supplied
        times = supplied / "measurement_kernel_times.csv"
    else:
        times = supplied
        profile_dir = supplied.parent
    if not times.is_file():
        raise SystemExit(f"kernel timing CSV not found: {times}")
    return profile_dir.resolve(), times.resolve()


def load_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"Agent_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp", "DurationNs"}
    missing = required.difference(rows[0] if rows else ())
    if missing:
        raise SystemExit(f"{path} is empty or missing columns: {', '.join(sorted(missing))}")
    return rows


def agent_number(agent: str) -> int:
    match = re.search(r"\d+", agent)
    return int(match.group()) if match else sys.maxsize


def parse_numbers(value: str, option: str) -> set[int]:
    try:
        return {int(item.strip()) for item in value.split(",") if item.strip()}
    except ValueError as error:
        raise SystemExit(f"{option} expects comma-separated integers; got {value!r}") from error


def filter_rows(
    rows: list[dict[str, str]], args: argparse.Namespace, *, apply_kernel: bool = True
) -> tuple[list[dict[str, str]], dict[str, int]]:
    agents = sorted({row["Agent_Id"] for row in rows}, key=agent_number)
    rank_for_agent = {agent: rank for rank, agent in enumerate(agents)}
    if args.agent:
        wanted_agents = parse_numbers(args.agent, "--agent")
        selected = [row for row in rows if agent_number(row["Agent_Id"]) in wanted_agents]
    elif args.rank != "all":
        wanted_ranks = parse_numbers(args.rank, "--rank")
        unknown = wanted_ranks.difference(rank_for_agent.values())
        if unknown:
            raise SystemExit(
                f"rank(s) not present: {sorted(unknown)}; available: "
                f"{sorted(rank_for_agent.values())}"
            )
        selected = [row for row in rows if rank_for_agent[row["Agent_Id"]] in wanted_ranks]
    else:
        selected = rows
    if args.kernel and apply_kernel:
        try:
            pattern = re.compile(args.kernel)
        except re.error as error:
            raise SystemExit(f"invalid --kernel regex: {error}") from error
        selected = [row for row in selected if pattern.search(row["Kernel_Name"])]
    if not selected:
        raise SystemExit("no kernel rows matched the requested filters")
    return selected, rank_for_agent


def short_name(name: str, args: argparse.Namespace) -> str:
    name = name.replace("\n", " ")
    if args.full_names or len(name) <= args.name_width:
        return name
    return name[: args.name_width - 3] + "..."


def selected_slice(
    rows: list[object], args: argparse.Namespace, *, default_limit: int = 40
) -> list[object]:
    limit = default_limit if args.limit is None else args.limit
    end = None if limit == 0 else args.offset + limit
    return rows[args.offset:end]


def rank_label(agent: str, ranks: dict[str, int]) -> str:
    return f"R{ranks[agent]}/A{agent_number(agent)}"


def print_mapping(ranks: dict[str, int], selected: list[dict[str, str]]) -> None:
    present = sorted({row["Agent_Id"] for row in selected}, key=agent_number)
    mapping = ", ".join(f"R{ranks[a]}={a}" for a in present)
    print(f"Rank map: {mapping} (logical ranks inferred from sorted GPU agents)")


def summary_view(
    rows: list[dict[str, str]], ranks: dict[str, int], args: argparse.Namespace
) -> None:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        owner = "all" if args.combine_ranks else row["Agent_Id"]
        grouped[(owner, row["Kernel_Name"])].append(int(row["DurationNs"]))

    result = []
    totals_by_owner = defaultdict(int)
    for (owner, _), durations in grouped.items():
        totals_by_owner[owner] += sum(durations)
    for (owner, name), durations in grouped.items():
        total = sum(durations)
        result.append(
            {
                "owner": owner,
                "name": name,
                "calls": len(durations),
                "total": total,
                "average": total / len(durations),
                "max": max(durations),
                "percent": 100 * total / totals_by_owner[owner],
            }
        )
    keys = {
        "total": lambda row: (-row["total"], row["name"]),
        "average": lambda row: (-row["average"], row["name"]),
        "max": lambda row: (-row["max"], row["name"]),
        "calls": lambda row: (-row["calls"], row["name"]),
        "name": lambda row: (row["name"],),
    }
    result.sort(key=keys[args.sort])

    print_mapping(ranks, rows)
    owner_header = "Ranks" if args.combine_ranks else "Rank/Agent"
    print(
        f"{owner_header:<11} {'Calls':>8} {'Total ms':>10} {'Avg us':>10} "
        f"{'Max us':>10} {'%GPU':>7}  Kernel"
    )
    print("-" * (73 + args.name_width))
    for row in selected_slice(result, args):
        owner = row["owner"]
        label = "combined" if owner == "all" else rank_label(owner, ranks)
        print(
            f"{label:<11} {row['calls']:>8} {row['total'] / 1e6:>10.3f} "
            f"{row['average'] / 1e3:>10.3f} {row['max'] / 1e3:>10.3f} "
            f"{row['percent']:>7.2f}  {short_name(row['name'], args)}"
        )
    if args.combine_ranks:
        print("Note: combined totals sum concurrent GPU time; they are not wall-clock time.")


def timeline_view(
    rows: list[dict[str, str]], ranks: dict[str, int], args: argparse.Namespace
) -> None:
    rows = sorted(rows, key=lambda row: (int(row["Start_Timestamp"]), agent_number(row["Agent_Id"])))
    origin = int(rows[0]["Start_Timestamp"])
    print_mapping(ranks, rows)
    print(f"{'Seq':>7} {'Start ms':>11} {'Dur us':>10} {'Rank/Agent':<11} {'Queue':>7}  Kernel")
    print("-" * (62 + args.name_width))
    sliced = selected_slice(rows, args)
    for sequence, row in enumerate(sliced, start=args.offset):
        start_ms = (int(row["Start_Timestamp"]) - origin) / 1e6
        duration_us = int(row["DurationNs"]) / 1e3
        print(
            f"{sequence:>7} {start_ms:>11.3f} {duration_us:>10.3f} "
            f"{rank_label(row['Agent_Id'], ranks):<11} {row.get('Queue_Id', ''):>7}  "
            f"{short_name(row['Kernel_Name'], args)}"
        )
    print("Rows are ordered by GPU start timestamp; different agents/queues may overlap.")


def load_model_config(profile_dir: pathlib.Path, args: argparse.Namespace) -> tuple[pathlib.Path | None, dict]:
    candidates: list[pathlib.Path] = []
    if args.model_config:
        candidates.append(pathlib.Path(args.model_config).expanduser())
    metadata = profile_dir / "metadata.txt"
    if metadata.is_file():
        for line in metadata.read_text().splitlines():
            if line.startswith("model="):
                candidates.append(pathlib.Path(line.split("=", 1)[1]) / "config.json")
    if os.environ.get("MODEL"):
        candidates.append(pathlib.Path(os.environ["MODEL"]) / "config.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve(), json.loads(candidate.read_text())
    return None, {}


def repeating_unit(values: list[str]) -> list[str]:
    for width in range(1, len(values) + 1):
        if all(value == values[index % width] for index, value in enumerate(values)):
            return values[:width]
    return values


def type_label(value: str) -> str:
    lowered = value.lower()
    if "sliding" in lowered:
        return "SW"
    if "full" in lowered:
        return "FULL"
    return value


def pattern_view(
    profile_dir: pathlib.Path,
    rows: list[dict[str, str]],
    ranks: dict[str, int],
    args: argparse.Namespace,
) -> None:
    try:
        marker = re.compile(args.layer_marker)
        step_marker = re.compile(args.step_marker)
        output_filter = re.compile(args.kernel) if args.kernel else None
    except re.error as error:
        raise SystemExit(f"invalid pattern-view regex: {error}") from error
    path, config = load_model_config(profile_dir, args)
    layer_types = list(config.get("layer_types") or [])
    layers = args.layers or config.get("num_hidden_layers") or len(layer_types)
    if not layers:
        raise SystemExit("layer count unknown; provide --layers or --model-config")
    layers = int(layers)
    if layer_types and len(layer_types) != layers:
        raise SystemExit(f"model config has {len(layer_types)} layer types but {layers} layers")
    if not layer_types:
        layer_types = ["unknown"] * layers

    labels = [type_label(value) for value in layer_types]
    unit = repeating_unit(labels)
    repeat = layers // len(unit) if len(unit) and layers % len(unit) == 0 else 1
    print_mapping(ranks, rows)
    print(f"Model config: {path or '<not found; layer types unknown>'}")
    print(f"Layers: {layers}; pattern: {' -> '.join(unit)}" + (f" repeated {repeat} times" if repeat > 1 else ""))
    if config.get("sliding_window") is not None:
        print(f"Sliding-window size: {config['sliding_window']} tokens")
    print(f"Layer marker: /{args.layer_marker}/")
    print("Classification comes from config.json layer_types; the marker validates trace repetition.")

    for agent in sorted({row["Agent_Id"] for row in rows}, key=agent_number):
        agent_rows = sorted(
            (row for row in rows if row["Agent_Id"] == agent),
            key=lambda row: (int(row["Start_Timestamp"]), int(row["End_Timestamp"])),
        )
        row_index = {id(row): index for index, row in enumerate(agent_rows)}
        attention = [row for row in agent_rows if marker.search(row["Kernel_Name"])]
        steps, remainder = divmod(len(attention), layers)
        print()
        print(
            f"{rank_label(agent, ranks)}: {len(attention)} marker dispatches = "
            f"{steps} complete decode iterations x {layers} layers + {remainder} extra"
        )
        if not attention:
            continue
        if args.step is not None:
            if args.step < 0 or args.step >= steps:
                raise SystemExit(f"--step {args.step} unavailable for {rank_label(agent, ranks)}; range is 0..{steps - 1}")
            chunk = attention[args.step * layers : (args.step + 1) * layers]
            if args.markers_only:
                print(f"Decode iteration {args.step} attention markers:")
                print(f"{'Layer':>5} {'Type':<6} {'Dur us':>10} {'Start ms':>11}")
                origin = int(chunk[0]["Start_Timestamp"])
                for index, row in enumerate(chunk):
                    print(
                        f"{index:>5} {labels[index]:<6} {int(row['DurationNs']) / 1e3:>10.3f} "
                        f"{(int(row['Start_Timestamp']) - origin) / 1e6:>11.3f}"
                    )
                continue

            def find_step_start(first_layer_marker: dict[str, str]) -> int:
                marker_index = row_index[id(first_layer_marker)]
                floor = max(0, marker_index - 128)
                for index in range(marker_index - 1, floor - 1, -1):
                    if step_marker.search(agent_rows[index]["Kernel_Name"]):
                        return index
                return marker_index

            step_starts = [
                find_step_start(attention[index * layers]) for index in range(steps)
            ]
            step_start = step_starts[args.step]
            step_end = (
                step_starts[args.step + 1]
                if args.step + 1 < len(step_starts)
                else len(agent_rows)
            )

            marker_indices = [row_index[id(row)] for row in chunk]

            def find_layer_start(marker_index: int, floor: int) -> int:
                for index in range(marker_index - 1, max(floor, marker_index - 20) - 1, -1):
                    name = agent_rows[index]["Kernel_Name"]
                    if name == "_rmsnorm_kernel" or "_fused_add_rmsnorm_kernel" in name:
                        return index
                return marker_index

            layer_starts = []
            for layer, marker_index in enumerate(marker_indices):
                floor = step_start if layer == 0 else marker_indices[layer - 1] + 1
                layer_starts.append(find_layer_start(marker_index, floor))
            layer_widths = [
                right - left for left, right in zip(layer_starts, layer_starts[1:])
            ]
            typical_layer_rows = round(statistics.median(layer_widths)) if layer_widths else 0
            layers_end = min(step_end, layer_starts[-1] + typical_layer_rows)

            annotated = []
            for index in range(step_start, step_end):
                row = agent_rows[index]
                if index < layer_starts[0]:
                    phase = "PROLOGUE"
                elif index >= layers_end:
                    phase = "EPILOGUE"
                else:
                    layer = bisect.bisect_right(layer_starts, index) - 1
                    phase = f"L{layer:02}/{labels[layer]}"
                if output_filter and not output_filter.search(row["Kernel_Name"]):
                    continue
                annotated.append((index - step_start, phase, row))

            origin = int(agent_rows[step_start]["Start_Timestamp"])
            elapsed = (
                max(int(row["End_Timestamp"]) for row in agent_rows[step_start:step_end])
                - origin
            )
            print(
                f"Decode iteration {args.step}: {step_end - step_start} kernels, "
                f"{elapsed / 1e6:.3f} ms from first start to last end"
            )
            print(f"Step boundary marker: /{args.step_marker}/")
            print(
                f"{'Seq':>5} {'Start ms':>10} {'Dur us':>9} {'Phase':<10} "
                f"{'Queue':>6}  Kernel"
            )
            print("-" * (55 + args.name_width))
            for sequence, phase, row in selected_slice(
                annotated, args, default_limit=0
            ):
                print(
                    f"{sequence:>5} "
                    f"{(int(row['Start_Timestamp']) - origin) / 1e6:>10.3f} "
                    f"{int(row['DurationNs']) / 1e3:>9.3f} {phase:<10} "
                    f"{row.get('Queue_Id', ''):>6}  {short_name(row['Kernel_Name'], args)}"
                )
            print(
                "Phases are inferred from repeated norm/attention boundaries; "
                "timestamps remain the authoritative ordering."
            )
            continue

        complete = attention[: steps * layers]
        by_type: dict[str, list[int]] = defaultdict(list)
        by_layer: dict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(complete):
            layer = index % layers
            duration = int(row["DurationNs"])
            by_type[labels[layer]].append(duration)
            by_layer[layer].append(duration)
        print(f"{'Type':<8} {'Calls':>8} {'Avg us':>10} {'Min us':>10} {'Max us':>10}")
        for label in dict.fromkeys(labels):
            durations = by_type[label]
            if durations:
                print(
                    f"{label:<8} {len(durations):>8} {statistics.fmean(durations) / 1e3:>10.3f} "
                    f"{min(durations) / 1e3:>10.3f} {max(durations) / 1e3:>10.3f}"
                )
        print(f"{'Layer':>5} {'Type':<6} {'Calls':>8} {'Avg us':>10} {'Min us':>10} {'Max us':>10}")
        layer_rows = list(range(layers))
        for layer in selected_slice(layer_rows, args):
            durations = by_layer[layer]
            print(
                f"{layer:>5} {labels[layer]:<6} {len(durations):>8} "
                f"{statistics.fmean(durations) / 1e3:>10.3f} "
                f"{min(durations) / 1e3:>10.3f} {max(durations) / 1e3:>10.3f}"
            )


def main() -> int:
    args = arguments()
    profile_dir, times = resolve_profile(args.profile)
    all_rows = load_rows(times)
    rows, ranks = filter_rows(all_rows, args, apply_kernel=args.view != "pattern")
    print(f"Input: {times}")
    if args.view == "summary":
        summary_view(rows, ranks, args)
    elif args.view == "timeline":
        timeline_view(rows, ranks, args)
    else:
        pattern_view(profile_dir, rows, ranks, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
