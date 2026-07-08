# Speed Test Runbook — TalkNet vs LightASD

How to run the speed benchmark + tests yourself and read the real output.
Branch: `feat/in-memory-lightasd`. (Scratch note — not meant to be committed.)

> **Timing reality:** the shared fixture is a ~7-minute clip, so **one full pipeline
> run is ~3.5–4 min**. The full 5-run comparison is ~45–60 min. Ordered fastest→slowest
> below so you can start small.

---

## 0. One-time setup

```bash
cd /home/ubuntu/active-speaker-detection
uv sync --extra dev        # installs pytest into the env
# fetch the shared clip once (also auto-downloads on first test use):
uv run python -c "from utils.sample_fixture import ensure_sample; print(ensure_sample())"
```

## 1. Fast sanity — CPU unit tests (~2 sec)

```bash
uv run pytest tests/test_profiling.py tests/test_compare.py tests/test_sample_fixture.py -v
```
Expect `9 passed`. Checks the profiler math, agreement metrics, and fixture downloader — no GPU needed.

## 2. See ONE run's per-stage speed breakdown (~4 min each)

Clearest view of where time goes. Run each model, then read the JSON:

```bash
# LightASD
uv run python main.py --videoName sample --videoFolder tests/fixtures --asdModel lightasd --metadataOnly
uv run python -m json.tool tests/fixtures/sample/pywork/timings.json

# TalkNet
uv run python main.py --videoName sample --videoFolder tests/fixtures --asdModel talknet --metadataOnly
uv run python -m json.tool tests/fixtures/sample/pywork/timings.json
```

Each run also logs a live one-liner:
```
[timing] total 205.8s (2.04x realtime); asd 43.7s (21.2% of total)
```
`timings.json` gives every stage with `seconds`, `pct_of_total`, `fps`, `peak_mem_mb`.
**Watch the `asd` percentage** — that's the model's share of total time.

## 3. The head-to-head speed comparison (headline)

**Quick pass (~15 min, 1 timed run each after warmup):**
```bash
uv run python tools/benchmark.py --videoName sample --videoFolder tests/fixtures \
    --models talknet lightasd --runs 1 --warmup 1
```

**Full pass (~45–60 min, 5 runs each → medians + stdev):**
```bash
uv run python tools/benchmark.py --videoName sample --videoFolder tests/fixtures \
    --models talknet lightasd --runs 5 --warmup 1
```

Output — a table plus `benchmark_results.json` (numbers illustrative):
```
stage         talknet (med s)     lightasd (med s)
------------------------------------------------
preprocess              XX.X               XX.X     <- shared, ~equal
scene                    X.X                X.X     <- shared
detect                  XX.X               XX.X     <- shared (YOLO)
track                    X.X                X.X     <- shared
asd                    1XX.X               4X.X     <- the ONLY model difference
output                  XX.X               XX.X     <- shared
total_seconds          2XX.X              2XX.X

asd speedup talknet/lightasd: N.NNx
```

**What the data showed on our run:** `asd` is only **~21% of total**, so even a big model
speedup moves the total modestly — the shared decode/detect/track/output stages dominate.

> **Faster iteration:** drop your own short clip at `tests/fixtures/myclip.mp4` and use
> `--videoName myclip` (skips the 7-min fixture) to get numbers in seconds instead of minutes.

## 4. GPU integration tests (optional, slower)

```bash
# Tracks identical across models — validates the comparison is a clean join (~7 min)
uv run pytest tests/test_tracks_deterministic.py -v

# Cross-model agreement guardrail; -s shows the sign/argmax prints (~8 min)
uv run pytest tests/test_cross_model_agreement.py -v -s
#   -> argmax_agreement ~0.70 (asserted >= 0.60), sign_agreement ~0.34 (informational)
```

## 5. Activate the speed-regression gate (per-box baseline)

It **skips** until you capture a baseline on your machine (the ratio is hardware-portable,
absolute times aren't):

```bash
uv run python main.py --videoName sample --videoFolder tests/fixtures --asdModel talknet --metadataOnly
T=$(python -c "import json;print(json.load(open('tests/fixtures/sample/pywork/timings.json'))['stages']['asd']['seconds'])")
uv run python main.py --videoName sample --videoFolder tests/fixtures --asdModel lightasd --metadataOnly
L=$(python -c "import json;print(json.load(open('tests/fixtures/sample/pywork/timings.json'))['stages']['asd']['seconds'])")
python -c "import json;json.dump({'asd_ratio_talknet_over_lightasd': $T/$L}, open('tests/fixtures/speed_baseline.json','w'))"

uv run pytest tests/test_speed_regression.py -v    # now asserts the ratio within ±25%
```

---

## Notes / gotchas

- **`test_face_detection.py` and `test_phase1_regression.py` are now slow** (minutes): they
  run over the 7-min shared clip. `test_phase1_regression.py` also needs its
  `baseline_talknet_metadata.json` re-captured against this clip (it skips until then).
- Artifacts (`timings.json`, `benchmark_results.json`, `speed_baseline.json`, the clip) are
  already gitignored by the repo's blanket `*.json` / `*.mp4` rules.
- The two ASD models emit different score scales, so cross-model agreement is compared
  scale-invariantly: **argmax** (which face is the speaker) is the real signal; raw-score
  **sign** is confounded by offset and kept informational only.

Design details: `docs/superpowers/specs/2026-07-08-speed-benchmark-design.md`
Implementation plan: `docs/superpowers/plans/2026-07-08-speed-benchmark.md`
