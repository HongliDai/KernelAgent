#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
One-shot pipeline runner: extract → dispatch → compose.

Usage:
  python -m Fuser.pipeline \
    --problem /abs/path/to/kernelbench_problem.py \
    --extract-model gpt-5 \
    --dispatch-model o4-mini \
    [--dispatch-jobs 1] \
    --compose-model o4-mini \
    --workers 4 --max-iters 5 \
    --llm-timeout-s 1200 --run-timeout-s 1200 \
    --out-root ./.fuse \
    [--verify] [--compose-max-iters 5]

Writes all artifacts into the run directory created by the extractor. The final
composed kernel and composition summary live under <run_dir>/compose_out.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .subgraph_extractor import extract_subgraphs_to_json
from .dispatch_kernel_agent import run as dispatch_run
from .compose_end_to_end import compose
from .lightweight_profiler import LightweightProfiler
from triton_kernel_agent.platform_config import get_platform_choices


def run_pipeline(
    problem_path: Path,
    extract_model: str,
    dispatch_model: str | None,
    compose_model: str,
    dispatch_jobs: int | str,
    workers: int,
    max_iters: int,
    llm_timeout_s: int,
    run_timeout_s: int,
    out_root: Path | None = None,
    verify: bool = True,
    compose_max_iters: int = 5,
    target_platform: str = "cuda",
    test_timeout_s: int = 30,
    enable_profiler: bool = False,
    profile_output_dir: Path | None = None,
    profile_export_csv: bool = False,
) -> dict:
    pipeline_t0 = time.time()
    profiler = LightweightProfiler(
        enabled=enable_profiler,
        problem_path=str(problem_path),
        target_platform=target_platform,
        profile_output_dir=profile_output_dir,
        export_csv=profile_export_csv,
    )
    if enable_profiler:
        os.environ["KA_ENABLE_PROFILER"] = "1"

    # Select default KernelAgent model if not provided: prefer GPT-5 for Level 2/3
    if dispatch_model is None:
        pp = str(problem_path)
        is_l2 = (
            ("/KernelBench/KernelBench/level2/" in pp)
            or ("/KernelBench/level2/" in pp)
            or ("level2/" in pp)
        )
        is_l3 = (
            ("/KernelBench/KernelBench/level3/" in pp)
            or ("/KernelBench/level3/" in pp)
            or ("level3/" in pp)
        )
        if is_l2 or is_l3:
            dispatch_model = "gpt-5"
        else:
            dispatch_model = "o4-mini"

    # Step 1: extract
    extract_t0 = time.time()
    run_dir, subgraphs_path = extract_subgraphs_to_json(
        problem_path=problem_path,
        model_name=extract_model,
        workers=workers,
        max_iters=max_iters,
        llm_timeout_s=llm_timeout_s,
        run_timeout_s=run_timeout_s,
        target_platform=target_platform,
        profiler=profiler,
    )
    extract_t1 = time.time()
    if enable_profiler:
        profiler.attach_run_dir(Path(run_dir))
        profiler.export_env()
        profiler.emit(
            {
                "event_type": "stage",
                "stage_name": "extract.total",
                "local_exec_start_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(extract_t0)),
                "local_exec_end_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(extract_t1)),
                "local_exec_latency_ms": round((extract_t1 - extract_t0) * 1000.0, 3),
                "success": True,
            }
        )

    # Step 2: dispatch to KernelAgent
    out_dir = Path(run_dir) / "kernels_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Resolve dispatch concurrency (support "auto")
    jobs_val: int
    if isinstance(dispatch_jobs, str) and dispatch_jobs.strip().lower() == "auto":
        try:
            with Path(subgraphs_path).open("r", encoding="utf-8") as f:
                _items = json.load(f)
            jobs_val = max(1, int(len(_items))) if isinstance(_items, list) else 1
        except Exception:
            jobs_val = 1
    else:
        try:
            jobs_val = max(1, int(dispatch_jobs))
        except Exception:
            jobs_val = 1

    dispatch_t0 = time.time()
    summary_path = dispatch_run(
        subgraphs_path=Path(subgraphs_path),
        out_dir=out_dir,
        agent_model=dispatch_model,
        jobs=jobs_val,
        target_platform=target_platform,
        max_iters=max_iters,
        test_timeout_s=test_timeout_s,
    )
    dispatch_t1 = time.time()
    if enable_profiler:
        profiler.emit(
            {
                "event_type": "stage",
                "stage_name": "dispatch.total",
                "local_exec_start_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(dispatch_t0)),
                "local_exec_end_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(dispatch_t1)),
                "local_exec_latency_ms": round((dispatch_t1 - dispatch_t0) * 1000.0, 3),
                "success": True,
            }
        )

    # Step 3: compose end-to-end
    compose_out = Path(run_dir) / "compose_out"
    compose_out.mkdir(parents=True, exist_ok=True)
    compose_t0 = time.time()
    comp_res = compose(
        problem_path=problem_path,
        subgraphs_path=Path(subgraphs_path),
        kernels_summary_path=summary_path,
        out_dir=compose_out,
        model_name=compose_model,
        verify=verify,
        max_iters=compose_max_iters,
        target_platform=target_platform,
    )
    compose_t1 = time.time()

    result = {
        "run_dir": str(run_dir),
        "subgraphs": str(subgraphs_path),
        "kernels_summary": str(summary_path),
        "composition": comp_res,
    }
    if enable_profiler:
        profiler.emit(
            {
                "event_type": "stage",
                "stage_name": "compose",
                "local_exec_start_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(compose_t0)),
                "local_exec_end_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(compose_t1)),
                "local_exec_latency_ms": round((compose_t1 - compose_t0) * 1000.0, 3),
                "compose_latency_ms": round((compose_t1 - compose_t0) * 1000.0, 3),
                "success": bool(comp_res.get("success")),
                "error_type": None if comp_res.get("success") else comp_res.get("verify_reason"),
            }
        )
        try:
            with Path(subgraphs_path).open("r", encoding="utf-8") as f:
                sg = json.load(f)
            n_subgraphs = len(sg) if isinstance(sg, list) else None
        except Exception:
            n_subgraphs = None
        pipeline_t1 = time.time()
        profiler.emit(
            {
                "event_type": "run_summary",
                "stage_name": "pipeline.total",
                "local_exec_start_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pipeline_t0)),
                "local_exec_end_ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pipeline_t1)),
                "local_exec_latency_ms": round((pipeline_t1 - pipeline_t0) * 1000.0, 3),
                "number_of_subgraphs": n_subgraphs,
                "number_of_workers": workers,
                "final_status": "success" if comp_res.get("success") else "failed",
                "final_output_path": comp_res.get("composed_path"),
                "success": bool(comp_res.get("success")),
            }
        )
        summary = profiler.finalize(
            {
                "final_status": "success" if comp_res.get("success") else "failed",
                "final_output_path": comp_res.get("composed_path"),
            }
        )
        if profiler.paths is not None:
            result["profiling"] = {
                "events": str(profiler.paths.events_path),
                "summary": str(profiler.paths.summary_path),
                "csv": str(profiler.paths.csv_path) if profile_export_csv else None,
                "aggregate": summary,
            }

    return result


def main(argv: list[str] | None = None) -> int:
    # Load .env if present for OPENAI_API_KEY, proxies, etc.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: extract → dispatch → compose"
    )
    p.add_argument("--problem", required=True, help="Absolute path to the problem file")
    p.add_argument("--extract-model", default="gpt-5")
    p.add_argument(
        "--dispatch-model",
        default=None,
        help="KernelAgent model (default: gpt-5 for level2 problems, else o4-mini)",
    )
    p.add_argument(
        "--dispatch-jobs",
        type=str,
        default="2",
        help="Max concurrent KernelAgent subgraph tasks (default: 2); use 'auto' to match subgraph count",
    )
    p.add_argument("--compose-model", default="o4-mini")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-iters", type=int, default=5, help="Extractor iter budget")
    p.add_argument("--llm-timeout-s", type=int, default=1200)
    p.add_argument("--run-timeout-s", type=int, default=1200)
    p.add_argument("--out-root", default=None)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--compose-max-iters", type=int, default=5)
    p.add_argument(
        "--target-platform",
        default="cuda",
        choices=get_platform_choices(),
        help="Target platform",
    )
    p.add_argument("--test-timeout-s", type=int, default=30)
    p.add_argument("--enable-profiler", action="store_true")
    p.add_argument(
        "--profile-output-dir",
        default=None,
        help="Optional profiler output dir (default: .fuse/<run_id>/profiling)",
    )
    p.add_argument("--profile-export-csv", action="store_true")
    args = p.parse_args(argv)

    problem_path = Path(args.problem).resolve()
    if not problem_path.is_file():
        print(f"problem not found: {problem_path}")
        return 2

    try:
        res = run_pipeline(
            problem_path=problem_path,
            extract_model=args.extract_model,
            dispatch_model=args.dispatch_model,
            compose_model=args.compose_model,
            dispatch_jobs=args.dispatch_jobs,
            workers=args.workers,
            max_iters=args.max_iters,
            llm_timeout_s=args.llm_timeout_s,
            run_timeout_s=args.run_timeout_s,
            out_root=Path(args.out_root) if args.out_root else None,
            verify=args.verify,
            compose_max_iters=args.compose_max_iters,
            target_platform=args.target_platform,
            test_timeout_s=args.test_timeout_s,
            enable_profiler=args.enable_profiler,
            profile_output_dir=Path(args.profile_output_dir).resolve()
            if args.profile_output_dir
            else None,
            profile_export_csv=args.profile_export_csv,
        )
        print(json.dumps(res, indent=2))
        return 0
    except SystemExit as e:
        try:
            return int(e.code) if e.code is not None else 1
        except Exception:
            try:
                import sys as _sys

                print(str(e), file=_sys.stderr)
            except Exception:
                pass
            return 1
    except Exception as e:
        print(f"pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
