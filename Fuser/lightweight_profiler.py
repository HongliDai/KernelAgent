from __future__ import annotations

import csv
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl  # Linux/Unix advisory lock
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


ENV_ENABLE = "KA_ENABLE_PROFILER"
ENV_EVENTS_PATH = "KA_PROFILE_EVENTS_PATH"
ENV_SUMMARY_PATH = "KA_PROFILE_SUMMARY_PATH"
ENV_CSV_PATH = "KA_PROFILE_CSV_PATH"
ENV_RUN_ID = "KA_PROFILE_RUN_ID"
ENV_PROBLEM_PATH = "KA_PROFILE_PROBLEM_PATH"
ENV_TARGET_PLATFORM = "KA_PROFILE_TARGET_PLATFORM"
ENV_SUPPORTS_USAGE = "KA_PROFILE_SUPPORTS_USAGE"


_DEF_EVENT_FIELDS = [
    "event_type",
    "stage_name",
    "worker_id",
    "iteration_id",
    "model_name",
    "request_start_ts",
    "request_end_ts",
    "llm_latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "success",
    "error_type",
    "local_exec_start_ts",
    "local_exec_end_ts",
    "local_exec_latency_ms",
    "verify_latency_ms",
    "compose_latency_ms",
    "subprocess_exit_code",
    "timeout_flag",
    "run_id",
    "problem_path",
    "target_platform",
    "number_of_subgraphs",
    "number_of_workers",
    "total_rounds",
    "final_status",
    "final_output_path",
    "provider_name",
    "provider_supports_usage",
    "timestamp",
    "event_id",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_ms(delta_s: float) -> float:
    return round(delta_s * 1000.0, 3)


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def extract_usage_tokens(usage: dict[str, Any] | None) -> tuple[int | None, int | None, int | None]:
    if not usage:
        return None, None, None
    prompt = _safe_int(usage.get("prompt_tokens"))
    if prompt is None:
        prompt = _safe_int(usage.get("input_tokens"))

    completion = _safe_int(usage.get("completion_tokens"))
    if completion is None:
        completion = _safe_int(usage.get("output_tokens"))

    total = _safe_int(usage.get("total_tokens"))
    if total is None:
        if prompt is not None and completion is not None:
            total = prompt + completion
        else:
            # nested structures seen in some SDK responses
            in_tok = None
            out_tok = None
            if isinstance(usage.get("input_token_details"), dict):
                in_tok = _safe_int(usage["input_token_details"].get("total"))
            if isinstance(usage.get("output_token_details"), dict):
                out_tok = _safe_int(usage["output_token_details"].get("total"))
            if prompt is None:
                prompt = in_tok
            if completion is None:
                completion = out_tok
            if prompt is not None and completion is not None:
                total = prompt + completion
    return prompt, completion, total


@dataclass
class ProfilerPaths:
    profiling_dir: Path
    events_path: Path
    summary_path: Path
    csv_path: Path


class LightweightProfiler:
    def __init__(
        self,
        enabled: bool,
        run_id: str | None = None,
        problem_path: str | None = None,
        target_platform: str | None = None,
        profile_output_dir: Path | None = None,
        export_csv: bool = False,
    ) -> None:
        self.enabled = enabled
        self.run_id = run_id
        self.problem_path = problem_path
        self.target_platform = target_platform
        self.profile_output_dir = profile_output_dir
        self.export_csv = export_csv
        self._paths: ProfilerPaths | None = None
        self._pending_events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def paths(self) -> ProfilerPaths | None:
        return self._paths

    def attach_run_dir(self, run_dir: Path) -> None:
        if not self.enabled:
            return
        run_id = run_dir.name
        self.run_id = run_id
        if self.profile_output_dir is None:
            profiling_dir = run_dir / "profiling"
        else:
            profiling_dir = self.profile_output_dir
        profiling_dir.mkdir(parents=True, exist_ok=True)
        self._paths = ProfilerPaths(
            profiling_dir=profiling_dir,
            events_path=profiling_dir / "profile_events.jsonl",
            summary_path=profiling_dir / "profile_summary.json",
            csv_path=profiling_dir / "profile_events.csv",
        )
        self._flush_pending_locked()

    def export_env(self) -> None:
        if not self.enabled or not self._paths:
            return
        os.environ[ENV_ENABLE] = "1"
        os.environ[ENV_EVENTS_PATH] = str(self._paths.events_path)
        os.environ[ENV_SUMMARY_PATH] = str(self._paths.summary_path)
        os.environ[ENV_CSV_PATH] = str(self._paths.csv_path)
        if self.run_id:
            os.environ[ENV_RUN_ID] = self.run_id
        if self.problem_path:
            os.environ[ENV_PROBLEM_PATH] = self.problem_path
        if self.target_platform:
            os.environ[ENV_TARGET_PLATFORM] = self.target_platform

    def _write_jsonl_line(self, line: str) -> None:
        assert self._paths is not None
        p = self._paths.events_path
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                except Exception:
                    pass
            f.write(line)
            f.write("\n")
            if fcntl is not None:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

    def _flush_pending_locked(self) -> None:
        if not self._paths:
            return
        for ev in self._pending_events:
            self._write_jsonl_line(json.dumps(ev, ensure_ascii=False))
        self._pending_events.clear()

    def emit(self, event: dict[str, Any]) -> None:
        if not self.enabled:
            return
        base = {
            "timestamp": _utc_now_iso(),
            "event_id": uuid.uuid4().hex,
            "run_id": self.run_id,
            "problem_path": self.problem_path,
            "target_platform": self.target_platform,
        }
        base.update(event)
        with self._lock:
            if self._paths is None:
                self._pending_events.append(base)
                return
            self._write_jsonl_line(json.dumps(base, ensure_ascii=False))

    @contextmanager
    def stage_timer(
        self,
        stage_name: str,
        worker_id: str | None = None,
        iteration_id: int | str | None = None,
        event_type: str = "stage",
        **extra: Any,
    ):
        t0 = time.time()
        start_iso = _utc_now_iso()
        success = True
        error_type: str | None = None
        try:
            yield
        except Exception as e:
            success = False
            error_type = e.__class__.__name__
            raise
        finally:
            t1 = time.time()
            payload: dict[str, Any] = {
                "event_type": event_type,
                "stage_name": stage_name,
                "worker_id": worker_id,
                "iteration_id": str(iteration_id) if iteration_id is not None else None,
                "local_exec_start_ts": start_iso,
                "local_exec_end_ts": _utc_now_iso(),
                "local_exec_latency_ms": _to_ms(t1 - t0),
                "success": success,
                "error_type": error_type,
            }
            payload.update(extra)
            self.emit(payload)

    def finalize(self, aggregate_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.enabled or not self._paths:
            return {}
        events = read_events(self._paths.events_path)
        summary = build_summary(
            events=events,
            run_id=self.run_id,
            problem_path=self.problem_path,
            target_platform=self.target_platform,
        )
        if aggregate_overrides:
            summary.update(aggregate_overrides)
        self._paths.summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if self.export_csv:
            export_events_csv(events, self._paths.csv_path)
        return summary


def read_events(events_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not events_path.is_file():
        return out
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    return out


def export_events_csv(events: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(_DEF_EVENT_FIELDS)
    # add dynamic keys while preserving order
    dynamic_keys: list[str] = []
    seen = set(fields)
    for ev in events:
        for k in ev.keys():
            if k not in seen:
                dynamic_keys.append(k)
                seen.add(k)
    fields.extend(dynamic_keys)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ev in events:
            writer.writerow(ev)


def build_summary(
    events: list[dict[str, Any]],
    run_id: str | None,
    problem_path: str | None,
    target_platform: str | None,
) -> dict[str, Any]:
    stage_time_ms: dict[str, float] = {}
    stage_tokens: dict[str, dict[str, int]] = {}
    worker_rounds: dict[str, set[str]] = {}
    worker_latency: dict[str, float] = {}
    worker_tokens: dict[str, int] = {}
    retries_by_stage: dict[str, int] = {}

    num_subgraphs = None
    final_status = "unknown"
    final_output_path = None

    for ev in events:
        stage = str(ev.get("stage_name") or "")
        worker = ev.get("worker_id")
        iteration = ev.get("iteration_id")

        lat = ev.get("local_exec_latency_ms")
        if isinstance(lat, (int, float)) and stage:
            stage_time_ms[stage] = round(stage_time_ms.get(stage, 0.0) + float(lat), 3)
            if isinstance(worker, str):
                worker_latency[worker] = round(
                    worker_latency.get(worker, 0.0) + float(lat), 3
                )

        p = _safe_int(ev.get("prompt_tokens")) or 0
        c = _safe_int(ev.get("completion_tokens")) or 0
        t = _safe_int(ev.get("total_tokens")) or 0
        if stage and (p or c or t):
            tok = stage_tokens.setdefault(stage, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            tok["prompt_tokens"] += p
            tok["completion_tokens"] += c
            tok["total_tokens"] += t
            if isinstance(worker, str):
                worker_tokens[worker] = worker_tokens.get(worker, 0) + t

        if isinstance(worker, str) and iteration is not None:
            worker_rounds.setdefault(worker, set()).add(str(iteration))

        if ev.get("event_type") == "run_summary":
            if ev.get("number_of_subgraphs") is not None:
                num_subgraphs = ev.get("number_of_subgraphs")
            if ev.get("final_status"):
                final_status = str(ev.get("final_status"))
            if ev.get("final_output_path"):
                final_output_path = str(ev.get("final_output_path"))

    for stage, rounds in _stage_iteration_counter(events).items():
        retries_by_stage[stage] = max(0, rounds - 1)

    worker_slowest = sorted(worker_latency.items(), key=lambda x: x[1], reverse=True)[:3]
    worker_token_top = sorted(worker_tokens.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "run_id": run_id,
        "problem_path": problem_path,
        "target_platform": target_platform,
        "number_of_subgraphs": num_subgraphs,
        "number_of_workers": len(worker_rounds),
        "total_rounds": sum(len(v) for v in worker_rounds.values()),
        "final_status": final_status,
        "final_output_path": final_output_path,
        "stage_total_latency_ms": stage_time_ms,
        "stage_total_tokens": stage_tokens,
        "slowest_workers_top3": [
            {"worker_id": wid, "total_latency_ms": lat} for wid, lat in worker_slowest
        ],
        "token_heaviest_workers_top3": [
            {"worker_id": wid, "total_tokens": tok} for wid, tok in worker_token_top
        ],
        "retries_by_stage": retries_by_stage,
        "event_count": len(events),
    }


def _stage_iteration_counter(events: list[dict[str, Any]]) -> dict[str, int]:
    d: dict[str, set[str]] = {}
    for ev in events:
        stage = ev.get("stage_name")
        iteration = ev.get("iteration_id")
        if not stage or iteration is None:
            continue
        d.setdefault(str(stage), set()).add(str(iteration))
    return {k: len(v) for k, v in d.items()}


_profiler_singleton: LightweightProfiler | None = None


def get_profiler_from_env() -> LightweightProfiler | None:
    global _profiler_singleton
    if _profiler_singleton is not None:
        return _profiler_singleton
    if os.getenv(ENV_ENABLE, "0") != "1":
        return None
    events_path = os.getenv(ENV_EVENTS_PATH)
    if not events_path:
        return None
    p = LightweightProfiler(
        enabled=True,
        run_id=os.getenv(ENV_RUN_ID),
        problem_path=os.getenv(ENV_PROBLEM_PATH),
        target_platform=os.getenv(ENV_TARGET_PLATFORM),
        export_csv=False,
    )
    profiling_dir = Path(events_path).resolve().parent
    p._paths = ProfilerPaths(
        profiling_dir=profiling_dir,
        events_path=Path(events_path).resolve(),
        summary_path=Path(os.getenv(ENV_SUMMARY_PATH, profiling_dir / "profile_summary.json")),
        csv_path=Path(os.getenv(ENV_CSV_PATH, profiling_dir / "profile_events.csv")),
    )
    _profiler_singleton = p
    return p


def emit_event_from_env(event: dict[str, Any]) -> None:
    p = get_profiler_from_env()
    if p is None:
        return
    p.emit(event)
