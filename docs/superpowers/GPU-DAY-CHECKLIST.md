# GPU-Day Verification Checklist — in-memory Light-ASD branch

Branch: `feat/in-memory-lightasd` (17 commits off `master`). All 10 plan tasks + a
post-review fix wave are implemented and internally reviewed on a CPU-only machine.
**Do NOT merge until the GPU checks below pass** — the runtime path needs CUDA and
none of it has actually executed yet.

Setup: `uv sync` on the GPU box (resolves CUDA torch), put a short real clip with a
visible face + speech at `tests/fixtures/sample.mp4`.

## 1. Smoke run — both models
```
uv run python main.py --videoName sample --videoFolder tests/fixtures --asdModel lightasd
uv run python main.py --videoName sample --videoFolder tests/fixtures --asdModel talknet --metadataOnly
```
Expect: completes; NO `pyframes/` dir, NO per-track `.wav` under `pycrop/`;
`pyavi/video_out.avi` marks the correct speaker (green box); `pywork/*.json` written.
Watch the `[timing]` log lines — detect + asd stages should be far faster than master.

## 2. CRITICAL — confirm the visual pathway is actually alive (the whole-branch review's top risk)
The final review found visual crops were being fed to the model in `[0,1]` while the
frontends expect `[0,255]`; the fix (commit `f3ded20`) scales to `[0,255]` and is locked
by a CPU unit test (`tests/test_gpu_crop_features.py`). But behavioral proof needs a GPU +
**multi-face** clip (single-face clips can look right on audio alone):
- Run on a clip with 2+ visible faces where only one is speaking.
- Confirm the speaking face gets a high score and the silent face a low score — i.e. the
  model is using vision, not just audio.
- Quick instrumentation: assert `inputV.std() > 1.0` in `run_asd_inference_gpu` (a dead
  visual pathway would have near-zero variance).
- Do NOT trust the Phase-1 regression gate for this — it compares new-TalkNet vs
  master-TalkNet, both fed the same features, so it is structurally blind to this bug.

## 3. Phase-1 regression gate
- Capture a baseline first: `git checkout master`, run the pipeline on the fixture with
  `--metadataOnly`, copy `pywork/frame_metadata.json` →
  `tests/fixtures/baseline_talknet_metadata.json`, `git checkout -`.
- Run `uv run pytest tests/test_phase1_regression.py -v` → expect ≥0.99 speaking-decision
  agreement (new in-memory TalkNet vs master TalkNet). If it fails, check RGB→BGR ordering
  in detection and the MFCC slice boundaries.

## 4. Run the CUDA-gated unit test
`uv run pytest tests/test_lightasd_model.py -v` → forward-shape check on real CUDA
(score length == T, finite). Currently skipped on CPU.

## 5. Carried risks to eyeball (all reasoned-through, none expected to fire)
- **Score reshape** (`numpy.asarray(scores).reshape(len(batch), -1)`): proven correct for
  both models by review (mismatch would crash `forward_audio_visual_backend`, not corrupt).
  Confirm both models produce sane per-frame scores.
- **PyAV `num_frames` metadata**: `run_face_detection` is hardened (commit `3754407`) against
  metadata≠decoded-count, but confirm on your real clip that frame counts line up.
- **Frame-source alignment**: crop pass now decodes `args.videoFilePath` (25fps `video.avi`),
  same timeline as detection/tracks/viz (fixed from `args.videoPath`).

## 6. Weight domain note
Default is `pretrain_AVA_CVPR.model` (AVA). If your footage is talking-head/meeting, try
`--lightAsdWeights ./weights/lightasd/finetuning_TalkSet.model` and compare.

## After green
`docs/superpowers/plans/2026-07-04-in-memory-lightasd.md` (plan) and the design spec are the
reference. When checks pass, merge via the team's normal PR flow (squash).

## Deferred non-blocking polish (from reviews, safe to skip or do later)
- `test_phase1_regression.py`: unused `tmp_path`, no cleanup of `tests/fixtures/sample/`.
- Add unit tests for the Fix-2 (metadata mismatch) and Fix-3 (empty-score track) paths.
- `helpers.py` score-smoothing window uses `len(score)-1` upper bound (drops tail element;
  pre-existing, now yields 0.0 instead of NaN for single-score tracks).
- Hoist the `len(score)==0` check above the per-frame loop in `export_metadata`.
