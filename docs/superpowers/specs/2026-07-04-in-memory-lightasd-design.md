# In-Memory Streaming ASD with Light-ASD — Design

**Date:** 2026-07-04
**Status:** Approved-pending-review
**Goal:** Make the offline ASD pipeline fast by (1) eliminating disk I/O for frames and per-track audio, and (2) replacing TalkNet with the lighter Light-ASD model. Same offline batch behavior and output format; no real-time/streaming requirement.

## Decisions (locked)

| Question | Decision |
|----------|----------|
| Use case | Offline batch, faster (not real-time streaming) |
| Model | Swap TalkNet → **Light-ASD** (Junhua-Liao/Light-ASD, CVPR 2023, MIT) |
| Legacy disk path | **Remove entirely** — no fallback to `extract_frames` / `crop_video` / standard TalkNet |
| Default weights | **`pretrain_AVA.model`** (AVA-ActiveSpeaker). `finetuning_TalkSet.model` available via config for talking-head/meeting footage |

## Problem

Current pipeline decodes the video multiple times and round-trips through disk:

1. `extract_video` → re-encode to `pyavi/video.avi` (1 write) — *keep, normalizes to 25 fps*
2. `extract_audio` → `pyavi/audio.wav` (1 write) — *keep, single source of truth*
3. **`extract_frames` → `pyframes/%06d.jpg` (N writes)** — dominant cost, exists only to feed YOLO
4. **`run_face_detection` → `cv2.imread` ×N** — reads every JPEG back
5. **`extract_audio_only` → per-track `.wav` (T writes) + `wavfile.read` ×T** — per-track audio round-trip
6. `crop_video` → per-track `.avi` (standard path) — already bypassed by the GPU path

The GPU path (`run_talknet_inference_gpu`) already decodes in memory via `decode_video_chunked` for cropping. The JPEG detour and per-track WAV are pure waste. Model compute is **not** the primary bottleneck; disk I/O is. Light-ASD swap is a secondary compute win.

## Architecture

Two in-memory decode passes over `pyavi/video.avi` (both cheap vs. JPEG round-trip); no intermediate images or per-track audio on disk.

```
extract_video (avi, 25fps)  ──┐
extract_audio (audio.wav)   ──┤
                              │
Pass 1: decode_video_chunked ─┴─► YOLO (in-memory RGB batches) ─► dets
                                              │
scene_detect + track_shot ◄───────────────────┘
        │
        ▼  allTracks
Pass 2: decode_video_chunked ─► crop_faces_gpu ─► grayscale 112×112 features (RAM)
global MFCC (from audio.wav, sliced per track) ──────────────┘
        │
        ▼
Light-ASD forward (visual/audio frontends → av backend → lossAV) ─► scores
        │
        ▼
speaker indices → summary/metadata (+ optional viz decode pass if not --metadataOnly)
```

RAM stays bounded by `chunk_size` (frames per chunk), same mechanism the crop pass already uses.

## Phase 1 — Kill disk I/O (TalkNet unchanged; provable regression baseline)

Ship this first with TalkNet still active so scores can be proven byte-identical (within rounding) to the current pipeline before the model swap.

- **`main.py`**: remove `extract_frames(args)` call; remove `pyframesPath` creation in `prepare_paths`.
- **`model/yoloFace.py::run_face_detection`**: change source from `glob(pyframes/*.jpg)` + `cv2.imread` to iterating `decode_video_chunked(args.videoFilePath, chunk_size, device_id)`. Run YOLO batched on in-memory frames (RGB, from GPU tensor → the form ultralytics accepts). Emit the identical `dets` list keyed by absolute frame index. Frame indexing is preserved because the source is the 25 fps-normalized `video.avi`.
- **Audio → one global MFCC**: load `pyavi/audio.wav` once (16 kHz mono), compute a single `python_speech_features.mfcc(...)` for the whole track. In `run_talknet_inference_gpu`, replace `wavfile.read(crop_path + ".wav")` with a slice `global_mfcc[frame0*4 : (frameN+1)*4, :]` (MFCC runs at 100 fps = 4× the 25 fps video). Delete the `extract_audio_only` per-track `.wav` write and the audio `ProcessPoolExecutor` stage.
- **`utils/helpers.py::visualization`**: replace `pyframes/*.jpg` reads with an in-memory decode pass over `video.avi`. Only runs when `--metadataOnly` is off.

**Exit criterion:** on a test clip, scores match the pre-change TalkNet output within rounding; `pyframes/` and per-track `.wav` no longer created.

## Phase 2 — Swap TalkNet → Light-ASD

- **Vendor** Light-ASD model source into `model/lightASD/` (MIT-licensed). Add a thin wrapper `lightASD()` mirroring the `talkNet()` interface used at the call site: `loadParameters(path)`, `.eval()`, and `.model.forward_visual_frontend`, `.model.forward_audio_frontend`, `.model.forward_audio_visual_backend`, plus `.lossAV.forward`.
- **Weights**: add `pretrain_AVA.model` download to `download_weights` (source: Junhua-Liao/Light-ASD `weight/`), path configurable. Keep TalkNet download only if TalkNet still selectable (see Phase-1 vs final).
- **Forward-site edits** in the (renamed) inference function — only tensor reshaping + one dropped call:
  - Visual: `(B, T, 112, 112)` → unsqueeze channel → `(B, 1, T, 112, 112)`.
  - Audio: `(B, T*4, 13)` → `(B, 1, 13, T*4)`.
  - **Drop** `forward_cross_attention` — Light-ASD fuses inside `forward_audio_visual_backend(audio_embed, visual_embed)`.
  - `lossAV.forward(out, labels=None)` unchanged.
- **Normalization check (must verify before wiring):** read Light-ASD's visual encoder to confirm it normalizes 0–255 grayscale internally (as TalkNet does with `(x/255 - 0.4161)/0.1688`). If it expects pre-normalized input, adjust the grayscale prep accordingly.
- **Keep** the `durationSet = [1,2,3,4,5,6]` / `weights = [3,3,2,1,1,1]` multi-scale aggregation — Light-ASD's demo uses the same trick and downstream scoring expects per-frame scores.

## Phase 3 — Config, cleanup, testing

- **`config/args.py`**: add `--asdModel {talknet,lightasd}` (default `lightasd`) and `--lightAsdWeights` path. Remove `--jpegQscale` and any `pyframes`-related args once Phase 1 lands.
- **Remove legacy disk path entirely** (per decision): delete `run_talknet_inference_standard`, `crop_video`, `extract_frames`, `extract_audio_only` (replaced), `evaluate_network` / `evaluate_network_batched`, and now-unused helpers `utils/dataset.py`, `utils/video_loader.py`. Remove the GPU→standard `try/except` fallback in `main()` — the in-memory path becomes the only path.
- **CUDA assumption**: the in-memory path relies on `decode_video_chunked` / `crop_faces_gpu` (CUDA). With the disk fallback removed, CUDA is required. If unavailable, fail fast with a clear message rather than maintaining a second compute path. (Documented assumption; revisit only if CPU-only offline runs become a requirement.)
- **Testing:**
  1. Phase-1 regression: same clip, assert TalkNet scores unchanged within rounding.
  2. Phase-2 sanity: Light-ASD scores plausible on the Columbia demo clip (correct speaker gets green box).
  3. Timing: log wall-clock per stage before/after; expect the frame decode/detect and audio stages to drop sharply.

## Components & responsibilities

| Unit | Does | Depends on |
|------|------|-----------|
| `run_face_detection` (rewritten) | in-memory decode → YOLO → `dets` | `gpu_video.decode_video_chunked`, ultralytics |
| global-MFCC helper | one MFCC from `audio.wav`, per-track slicing | `python_speech_features`, `scipy.io.wavfile` |
| `crop_faces_gpu` (unchanged) | grayscale 112×112 features from in-memory frames | `gpu_crop` |
| `lightASD` wrapper (new) | TalkNet-shaped interface over Light-ASD | vendored `model/lightASD/` |
| inference driver (renamed from `run_talknet_inference_gpu`) | tie crops+MFCC → Light-ASD forward → scores | above |

## Risks

- **Light-ASD normalization mismatch** → wrong scores. Mitigated by the explicit pre-wiring verification step.
- **YOLO input form from GPU tensor** — ultralytics `predict` accepts numpy/list; may need `.cpu().numpy()` per chunk, costing a device round-trip. Measure; if costly, detect on CPU-side numpy frames from the same decode.
- **Domain mismatch** of `pretrain_AVA.model` vs. actual footage (if talking-head/meeting) → offer `finetuning_TalkSet.model` via config.
- **No CPU compute fallback** after legacy removal — acceptable given GPU deployment; documented.

## Out of scope

Real-time/streaming inference, rolling-buffer transformer (Apple-style), face–voice association (SL-ASD), and any change to tracking (`track_shot`) or scene detection.
