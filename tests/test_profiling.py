from utils.profiling import Profiler, build_timing_report, aggregate_runs


def test_profiler_records_stage_and_zero_mem_on_cpu():
    p = Profiler()
    with p.stage("work"):
        sum(range(10_000))
    assert "work" in p.timings
    assert p.timings["work"] >= 0.0
    # No CUDA in the unit-test env -> peak mem recorded as 0.
    assert p.peak_mem["work"] == 0


def test_build_timing_report_pct_and_xrealtime():
    report = build_timing_report(
        timings={"a": 1.0, "b": 3.0},
        peak_mem={"a": 0, "b": 0},
        total_frames=200,
        fps=25,
    )
    assert report["total_seconds"] == 4.0
    assert report["stages"]["a"]["pct_of_total"] == 25.0
    assert report["stages"]["b"]["pct_of_total"] == 75.0
    # clip = 200/25 = 8s over 4s of processing -> 2x realtime
    assert report["end_to_end_xrealtime"] == 2.0


def test_aggregate_runs_median_min():
    reports = [
        build_timing_report({"a": s}, {"a": 0}, 100, 25) for s in (2.0, 4.0, 6.0)
    ]
    agg = aggregate_runs(reports)
    assert agg["n"] == 3
    assert agg["total_seconds"]["median"] == 4.0
    assert agg["total_seconds"]["min"] == 2.0
    assert agg["stages"]["a"]["median"] == 4.0
