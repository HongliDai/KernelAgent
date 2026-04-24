#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Fuser.lightweight_profiler import build_summary, read_events


def _print_table(title: str, rows: list[tuple[str, float | int]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (empty)")
        return
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k.ljust(width)}  {v}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Summarize KernelAgent profiler JSONL events")
    p.add_argument("--events", required=True, help="Path to profile_events.jsonl")
    p.add_argument("--summary-json", default=None, help="Optional output JSON path")
    args = p.parse_args(argv)

    events_path = Path(args.events).resolve()
    if not events_path.is_file():
        print(f"events file not found: {events_path}")
        return 2

    events = read_events(events_path)
    summary = build_summary(events, run_id=None, problem_path=None, target_platform=None)

    stage_latency = sorted(
        summary.get("stage_total_latency_ms", {}).items(), key=lambda x: x[1], reverse=True
    )
    stage_tokens = sorted(
        ((k, v.get("total_tokens", 0)) for k, v in summary.get("stage_total_tokens", {}).items()),
        key=lambda x: x[1],
        reverse=True,
    )
    slowest_workers = [
        (str(x.get("worker_id")), float(x.get("total_latency_ms", 0.0)))
        for x in summary.get("slowest_workers_top3", [])
    ]
    token_workers = [
        (str(x.get("worker_id")), int(x.get("total_tokens", 0)))
        for x in summary.get("token_heaviest_workers_top3", [])
    ]
    retry_stages = sorted(
        summary.get("retries_by_stage", {}).items(), key=lambda x: x[1], reverse=True
    )

    _print_table("每个阶段总耗时(ms):", [(k, round(v, 3)) for k, v in stage_latency])
    _print_table("每个阶段总 tokens:", stage_tokens)
    _print_table("最慢的 3 个 worker:", slowest_workers)
    _print_table("token 最多的 3 个 worker:", token_workers)
    _print_table("retry 次数最多的阶段:", retry_stages[:3])

    if args.summary_json:
        out = Path(args.summary_json).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsummary json: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
