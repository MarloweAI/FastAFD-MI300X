#!/usr/bin/env python3
"""Merge a same-node InferenceX reproduction with the published point table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-csv", type=Path, required=True)
    parser.add_argument("--reproduction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.published_csv.open(newline="", encoding="utf-8") as handle:
        published = {
            (int(row["tp"]), int(row["concurrency"])): row
            for row in csv.DictReader(handle)
        }

    rows: list[dict[str, str]] = []
    for point_dir in sorted(args.reproduction_root.glob("tp*-c*")):
        aggregates = list(point_dir.glob("agg_*.json"))
        if len(aggregates) != 1:
            raise SystemExit(f"expected one aggregate in {point_dir}, found {aggregates}")
        blob = json.loads(aggregates[0].read_text(encoding="utf-8"))
        key = (int(blob["tp"]), int(blob["conc"]))
        if key not in published:
            raise SystemExit(f"same-node point {key} is absent from {args.published_csv}")
        old = published[key]
        pub_tput = float(old["published_total_tput_per_gpu"])
        pub_out = float(old["published_output_tput_per_gpu"])
        pub_int = float(old["published_median_interactivity"])
        pub_ttft = float(old["published_median_ttft_ms"])
        new_tput = float(blob["tput_per_gpu"])
        new_out = float(blob["output_tput_per_gpu"])
        new_int = float(blob["median_intvty"])
        new_ttft = float(blob["median_ttft"]) * 1000.0

        def delta(new: float, old_value: float) -> str:
            return f"{100.0 * (new / old_value - 1.0):.3f}"

        rows.append(
            {
                "tp": str(key[0]),
                "concurrency": str(key[1]),
                "slurm_job_id": next(point_dir.glob("slurm-*.out")).stem.split("-")[-1],
                "published_total_tput_per_gpu": f"{pub_tput:.6f}",
                "reproduced_total_tput_per_gpu": f"{new_tput:.6f}",
                "total_tput_delta_pct": delta(new_tput, pub_tput),
                "published_output_tput_per_gpu": f"{pub_out:.6f}",
                "reproduced_output_tput_per_gpu": f"{new_out:.6f}",
                "published_median_interactivity": f"{pub_int:.6f}",
                "reproduced_median_interactivity": f"{new_int:.6f}",
                "interactivity_delta_pct": delta(new_int, pub_int),
                "published_median_ttft_ms": f"{pub_ttft:.6f}",
                "reproduced_median_ttft_ms": f"{new_ttft:.6f}",
                "ttft_delta_pct": delta(new_ttft, pub_ttft),
            }
        )

    if set(published) != {(int(r["tp"]), int(r["concurrency"])) for r in rows}:
        raise SystemExit("reproduction does not contain the complete published matrix")
    rows.sort(key=lambda row: (int(row["tp"]), int(row["concurrency"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
