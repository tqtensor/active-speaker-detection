"""Warmup + N-run speed benchmark comparing ASD models on one clip.

Usage:
    uv run python tools/benchmark.py \
        --videoName sample --videoFolder tests/fixtures \
        --models talknet lightasd --runs 5 --warmup 1
"""

import argparse
import json
import os
import subprocess
import sys

# `python tools/benchmark.py` puts this script's own directory (tools/) at
# sys.path[0], not the repo root, so the repo-root packages below (config,
# utils) would otherwise fail to import. main.py doesn't need this because it
# lives at the repo root itself. No editable install is registered (pyproject
# has `package = false`), so this is the only path to a reliable import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.logging_config import get_logger
from utils.profiling import aggregate_runs
from utils.sample_fixture import ensure_sample

logger = get_logger(__name__)


def _run_once(video_name, video_folder, model):
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "main.py",
            "--videoName",
            video_name,
            "--videoFolder",
            video_folder,
            "--asdModel",
            model,
            "--metadataOnly",
        ],
        check=True,
    )
    timings = os.path.join(video_folder, video_name, "pywork", "timings.json")
    with open(timings) as f:
        return json.load(f)


def benchmark_model(video_name, video_folder, model, runs, warmup):
    for _ in range(warmup):
        _run_once(video_name, video_folder, model)  # discarded
    reports = [_run_once(video_name, video_folder, model) for _ in range(runs)]
    return aggregate_runs(reports)


def _print_table(results):
    models = list(results)
    stages = list(results[models[0]]["stages"])
    header = f"{'stage':<12}" + "".join(f"{m + ' (med s)':>20}" for m in models)
    print(header)
    print("-" * len(header))
    for st in stages + ["total_seconds"]:
        row = f"{st:<12}"
        for m in models:
            val = (
                results[m]["stages"][st]["median"]
                if st != "total_seconds"
                else results[m]["total_seconds"]["median"]
            )
            row += f"{val:>20.3f}"
        print(row)
    if len(models) == 2 and results[models[0]]["stages"].get("asd"):
        a, b = models
        ra = results[a]["stages"]["asd"]["median"]
        rb = results[b]["stages"]["asd"]["median"]
        if rb:
            print(f"\nasd speedup {a}/{b}: {ra / rb:.2f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videoName", required=True)
    ap.add_argument("--videoFolder", required=True)
    ap.add_argument("--models", nargs="+", default=["talknet", "lightasd"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    if args.videoName == "sample":
        ensure_sample()  # ensure the shared public clip is present

    results = {
        m: benchmark_model(args.videoName, args.videoFolder, m, args.runs, args.warmup)
        for m in args.models
    }
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    _print_table(results)
    logger.info("Wrote benchmark_results.json")


if __name__ == "__main__":
    main()
