import statistics
import time
from contextlib import contextmanager

from config.logging_config import get_logger

logger = get_logger(__name__)

try:
    import torch

    _HAS_TORCH = True
except ImportError:  # torch is a prod dep; guard keeps the util importable anyway
    _HAS_TORCH = False


def _cuda_available():
    return _HAS_TORCH and torch.cuda.is_available()


class Profiler:
    """Collects per-stage wall-clock time and peak GPU memory.

    CUDA kernels are asynchronous, so a wall-clock region is only correct if it
    both starts and ends on a device sync. This profiler synchronizes at each
    stage boundary when CUDA is available; on CPU it degrades to plain timing
    with peak memory recorded as 0.
    """

    def __init__(self):
        self.timings = {}
        self.peak_mem = {}

    @contextmanager
    def stage(self, name):
        if _cuda_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if _cuda_available():
                torch.cuda.synchronize()
                self.peak_mem[name] = int(torch.cuda.max_memory_allocated())
            else:
                self.peak_mem[name] = 0
            self.timings[name] = time.perf_counter() - t0


def build_timing_report(timings, peak_mem, total_frames, fps):
    """Turn raw per-stage seconds + peak bytes into a normalized report dict."""
    total = sum(timings.values())
    clip_seconds = (total_frames / fps) if fps else 0.0
    stages = {}
    for name, sec in timings.items():
        stages[name] = {
            "seconds": round(sec, 3),
            "pct_of_total": round(100.0 * sec / total, 1) if total else 0.0,
            "fps": round(total_frames / sec, 1) if sec else 0.0,
            "peak_mem_mb": round(peak_mem.get(name, 0) / (1024 * 1024), 1),
        }
    return {
        "total_seconds": round(total, 3),
        "total_frames": total_frames,
        "fps": fps,
        "clip_seconds": round(clip_seconds, 3),
        "end_to_end_xrealtime": round(clip_seconds / total, 3) if total else 0.0,
        "stages": stages,
    }


def aggregate_runs(reports):
    """Aggregate N single-run reports into median/min/stdev per stage + total."""
    if not reports:
        raise ValueError("aggregate_runs requires at least one report")

    def stats(values):
        return {
            "median": round(statistics.median(values), 3),
            "min": round(min(values), 3),
            "stdev": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
        }

    stage_names = reports[0]["stages"].keys()
    return {
        "n": len(reports),
        "total_seconds": stats([r["total_seconds"] for r in reports]),
        "stages": {
            name: stats([r["stages"][name]["seconds"] for r in reports])
            for name in stage_names
        },
    }
