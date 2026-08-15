#!/usr/bin/env python3
"""Normalize AFD/InferenceX results and render the combined MI355X Pareto."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Point:
    system: str
    split: str
    concurrency: int
    throughput: float
    interactivity: float
    source: str
    frontier: bool = False
    split_frontier: bool = False


def is_frontier(point: Point, candidates: list[Point]) -> bool:
    return not any(
        other is not point
        and other.throughput >= point.throughput
        and other.interactivity >= point.interactivity
        and (
            other.throughput > point.throughput
            or other.interactivity > point.interactivity
        )
        for other in candidates
    )


def load_vllm(path: Path) -> list[Point]:
    points: list[Point] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = f"TP{row['tp']}"
            for system, prefix in (
                ("vLLM published", "published"),
                ("vLLM same-node", "reproduced"),
            ):
                points.append(
                    Point(
                        system=system,
                        split=label,
                        concurrency=int(row["concurrency"]),
                        # The source Pareto is normalized per GPU. Scale every
                        # system to an eight-MI355X node-equivalent total so the
                        # AFD full-node measurements share one honest x-axis.
                        throughput=8.0 * float(row[f"{prefix}_total_tput_per_gpu"]),
                        interactivity=float(row[f"{prefix}_median_interactivity"]),
                        source=str(path),
                    )
                )
    return points


def load_afd(root: Path) -> list[Point]:
    points: list[Point] = []
    split_pattern = re.compile(r"afd-(\d+)a-(\d+)f")
    for path in sorted(root.glob("afd-*a-*f/c*/*.json")):
        match = split_pattern.search(str(path))
        if match is None or path.name.startswith("agg_"):
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
            concurrency = int(blob["max_concurrency"])
            throughput = float(blob["total_token_throughput"])
            interactivity = 1000.0 / float(blob["median_tpot_ms"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        points.append(
            Point(
                system="FastAFD",
                split=f"{match.group(1)}:{match.group(2)}",
                concurrency=concurrency,
                throughput=throughput,
                interactivity=interactivity,
                source=str(path),
            )
        )
    if not points:
        raise SystemExit(f"no AFD benchmark JSON found below {root}")
    return points


def nice_max(value: float) -> float:
    target = max(value * 1.08, 1.0)
    magnitude = 10 ** math.floor(math.log10(target))
    scaled = target / magnitude
    step = 1 if scaled <= 1 else 2 if scaled <= 2 else 5 if scaled <= 5 else 10
    return step * magnitude


def write_csv(path: Path, points: list[Point]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "system",
                "split",
                "concurrency",
                "total_token_throughput_8gpu_node",
                "median_interactivity_output_tok_s",
                "global_frontier",
                "split_frontier",
                "source",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point.system,
                    point.split,
                    point.concurrency,
                    f"{point.throughput:.6f}",
                    f"{point.interactivity:.6f}",
                    str(point.frontier).lower(),
                    str(point.split_frontier).lower(),
                    point.source,
                ]
            )


def render_svg(path: Path, points: list[Point]) -> None:
    width, height = 1280, 760
    left, right, top, bottom = 112, 56, 108, 98
    plot_w, plot_h = width - left - right, height - top - bottom
    x_max = nice_max(max(point.throughput for point in points))
    y_max = nice_max(max(point.interactivity for point in points))

    def sx(value: float) -> float:
        return left + value / x_max * plot_w

    def sy(value: float) -> float:
        return top + plot_h - value / y_max * plot_h

    colors = {
        "7:1": "#d73027",
        "6:2": "#f46d43",
        "5:3": "#fdae61",
        "4:4": "#66bd63",
        "3:5": "#1a9850",
        "2:6": "#3288bd",
        "1:7": "#5e4fa2",
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<title>GPT-OSS-120B FastAFD and vLLM Pareto on MI355X</title>",
        "<desc>Higher and farther right is better. Dominated AFD points are faint.</desc>",
        "<style>text{font-family:Inter,system-ui,sans-serif;fill:#17212b}.grid{stroke:#dce3e8;stroke-width:1}.axis{stroke:#536471;stroke-width:1.5}.tick{font-size:13px;fill:#536471}.label{font-size:15px;font-weight:600}.small{font-size:12px}.splitline{fill:none;stroke-width:1.4;stroke-dasharray:4 4;opacity:.7}.envelope{fill:none;stroke:#111827;stroke-width:3}</style>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="112" y="38" font-size="24" font-weight="700">GPT-OSS-120B · 8K input / 1K output · MI355X</text>',
        '<text x="112" y="66" font-size="14" fill="#536471">FastAFD role splits versus original and same-node vLLM · higher and farther right is better</text>',
    ]
    for index in range(11):
        xv = x_max * index / 10
        px = sx(xv)
        svg.append(f'<line class="grid" x1="{px:.1f}" y1="{top}" x2="{px:.1f}" y2="{top + plot_h}"/>')
        svg.append(f'<text class="tick" x="{px:.1f}" y="{top + plot_h + 25}" text-anchor="middle">{xv / 1000:.0f}k</text>')
        yv = y_max * index / 10
        py = sy(yv)
        svg.append(f'<line class="grid" x1="{left}" y1="{py:.1f}" x2="{left + plot_w}" y2="{py:.1f}"/>')
        svg.append(f'<text class="tick" x="{left - 14}" y="{py + 5:.1f}" text-anchor="end">{yv:.0f}</text>')
    svg.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>',
            f'<text class="label" x="{left + plot_w / 2:.1f}" y="{height - 27}" text-anchor="middle">Total token throughput per 8-GPU node (tokens/s)</text>',
            f'<text class="label" transform="translate(27 {top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">Median interactivity (output tokens/s)</text>',
        ]
    )

    afd = [point for point in points if point.system == "FastAFD"]
    for split, color in colors.items():
        frontier = sorted(
            (point for point in afd if point.split == split and point.split_frontier),
            key=lambda point: point.throughput,
        )
        if len(frontier) > 1:
            coords = " ".join(f"{sx(p.throughput):.1f},{sy(p.interactivity):.1f}" for p in frontier)
            svg.append(f'<polyline class="splitline" stroke="{color}" points="{coords}"/>')

    envelope = sorted((point for point in afd if point.frontier), key=lambda p: p.throughput)
    if len(envelope) > 1:
        coords = " ".join(f"{sx(p.throughput):.1f},{sy(p.interactivity):.1f}" for p in envelope)
        svg.append(f'<polyline class="envelope" points="{coords}"/>')

    for point in points:
        px, py = sx(point.throughput), sy(point.interactivity)
        if point.system == "vLLM published":
            svg.append(f'<rect x="{px - 5:.1f}" y="{py - 5:.1f}" width="10" height="10" fill="#fff" stroke="#6b7280" stroke-width="2"/>')
        elif point.system == "vLLM same-node":
            svg.append(f'<path d="M {px:.1f} {py - 6:.1f} L {px + 6:.1f} {py:.1f} L {px:.1f} {py + 6:.1f} L {px - 6:.1f} {py:.1f} Z" fill="#00a6a6" stroke="#006d77"/>')
        else:
            opacity = "1" if point.split_frontier else ".22"
            color = colors[point.split]
            svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" stroke="{color}" opacity="{opacity}"/>')
            if point.frontier:
                label = html.escape(f"{point.split}/c{point.concurrency}")
                svg.append(f'<text class="small" x="{px + 8:.1f}" y="{py - 8:.1f}">{label}</text>')

    legend_x, legend_y = 690, 88
    svg.append(f'<rect x="{legend_x - 16}" y="{legend_y - 17}" width="550" height="32" rx="5" fill="#fff" stroke="#dce3e8"/>')
    svg.append(f'<rect x="{legend_x}" y="{legend_y - 6}" width="10" height="10" fill="#fff" stroke="#6b7280" stroke-width="2"/><text class="small" x="{legend_x + 17}" y="{legend_y + 4}">published vLLM</text>')
    svg.append(f'<path d="M {legend_x + 145} {legend_y - 7} l 7 7 -7 7 -7 -7 Z" fill="#00a6a6"/><text class="small" x="{legend_x + 158}" y="{legend_y + 4}">same-node vLLM</text>')
    svg.append(f'<circle cx="{legend_x + 285}" cy="{legend_y}" r="5" fill="#3288bd"/><text class="small" x="{legend_x + 296}" y="{legend_y + 4}">AFD split points</text>')
    svg.append(f'<line x1="{legend_x + 405}" y1="{legend_y}" x2="{legend_x + 438}" y2="{legend_y}" class="envelope"/><text class="small" x="{legend_x + 444}" y="{legend_y + 4}">AFD envelope</text>')
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--afd-root", type=Path, required=True)
    parser.add_argument("--vllm-csv", type=Path, required=True)
    parser.add_argument("--output-svg", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    points = load_vllm(args.vllm_csv) + load_afd(args.afd_root)
    afd = [point for point in points if point.system == "FastAFD"]
    for point in afd:
        point.frontier = is_frontier(point, afd)
        point.split_frontier = is_frontier(
            point, [candidate for candidate in afd if candidate.split == point.split]
        )
    for system in ("vLLM published", "vLLM same-node"):
        candidates = [point for point in points if point.system == system]
        for point in candidates:
            point.split_frontier = is_frontier(point, candidates)
    write_csv(args.output_csv, points)
    render_svg(args.output_svg, points)


if __name__ == "__main__":
    main()
