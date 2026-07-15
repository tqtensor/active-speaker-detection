# ASD Speed Bottleneck — Deep Dive & Path to 12× Realtime

**Date:** 2026-07-15
**Branch:** `feat/in-memory-lightasd`
**Hardware:** NVIDIA A10 (23 GB), lambda.ai
**Status:** Diagnosis confirmed on hardware, incl. a full 102-min HD run. Fix not yet implemented.

---

## TL;DR

The pipeline is **CPU-decode-bound, not model-bound and not GPU-bound.** During the
heaviest stage the A10 sits at **0% utilization**. The active-speaker model
(TalkNet / Light-ASD) — long assumed to be the bottleneck — is **~9 seconds, ~4% of
total**. The real cost is decoding the video (repeatedly) on a single CPU core and
running YOLO face detection on every frame.

Two beliefs going in were wrong; one was right:

| Belief                                                | Verdict       | Evidence                                                  |
| ----------------------------------------------------- | ------------- | --------------------------------------------------------- |
| TalkNet/Light-ASD sequential fusion is the bottleneck | ❌ **false**   | model forward = ~9 s, ~4% of total                        |
| A bigger GPU will fix it                              | ❌ **false**   | A10 at 0% `sm` for the entire 35-min detect stage         |
| "The read" (decode) is the bottleneck                 | ✅ **correct** | `dec 0` (NVDEC unused), one CPU core pinned, detect = 67% |

**12× realtime is achievable, but only architecturally** — strided detection + true
GPU-resident decode — never by swapping the model or buying a bigger card.

> **Reality check (added after the 102-min HD run):** the short 7-min fixture flattered
> us. On a real 102-min, higher-resolution clip the pipeline runs at **1.04× realtime**
> (98.5 min to process 102 min), and `detect` throughput *halves* (71.6 → 38.1 fps)
> because resolution is a first-class cost the code never controls for. The honest gap
> to 12× on real customer content is **~12×, not ~6×**, and closing it needs the *entire*
> fix stack landing — striding alone will not do it. See §8.

---

## Target

The incoming customer wants long-to-short fast: **1 hour of video processed in ~5
minutes = 12× realtime.** ASD is the major bottleneck component of long-to-short.

- Current (7-min fixture): **~2× realtime** (a 432 s clip takes ~221 s).
- Current (**102-min HD, the realistic baseline**): **1.04× realtime** (6133 s clip takes 5911 s).
- Required: **~12× realtime.** On real long content that is a **~12× improvement**, not 6×.
- Reference: Twelve Labs reportedly does <1 min on 30-min video (>30×), so 12× is
  physically real.

---

## Method

Per-stage timing comes from the CUDA-synchronized `Profiler` (`utils/profiling.py`),
emitted to `pywork/timings.json`. GPU utilization was read live with
`nvidia-smi` and `nvidia-smi dmon -s u`. All numbers below are metadata-only runs
(no visualization) so they represent the analysis path, not rendering.

---

## Findings

### 1. Per-stage breakdown (7-min fixture, Light-ASD, metadata-only)

Total **220.8 s**, end-to-end **1.96× realtime**:

| Stage      |   seconds | % of total |  fps | Note                                         |
| ---------- | --------: | ---------: | ---: | -------------------------------------------- |
| **detect** | **150.9** |  **68.3%** | 71.6 | YOLO on every frame + CPU roundtrip          |
| asd        |      39.7 |      18.0% |  272 | see breakdown below — only ~9 s is the model |
| preprocess |      20.8 |       9.4% |  520 | ffmpeg re-encode to `video.avi`              |
| scene      |       8.7 |       3.9% | 1240 | separate full CPU decode pass                |
| track      |      0.03 |       0.0% |    — | free                                         |
| output     |       0.7 |       0.3% |    — | metadata only                                |

### 2. The `detect` stage decomposed

Log timestamps expose the split inside detect and asd:

- Raw decode throughput (from the asd stage's GPU-resident decode+crop) ≈ **360 fps**
  → ~30 s to decode the whole clip.
- `detect` runs at only **71.6 fps**.

So of detect's 151 s, roughly **30 s is decode and ~121 s is YOLO forward + the
`.cpu().numpy()` / BGR roundtrip that feeds it.** 71.6 fps for YOLOv11-**nano** is
far below the several-hundred fps the model is capable of — the gap is the per-frame
CPU roundtrip and Python dispatch, not the network.

### 2a. The YOLO forward is already batched — and that's *why* batching didn't help

`run_face_detection` already passes a list of 32 frames to `model.predict`
(`model/yoloFace.py:52-54`), so the GPU **forward pass is batched.** This is the trap:
batching accelerated the one stage that was never the bottleneck. A nano-model forward
on a batch of 32 finishes in milliseconds. The 50 minutes is everything that happens
*around* the forward, still per-frame and still on the CPU:

| Per-frame work in detect                                                                       | Batched?       | On GPU?           |
| ---------------------------------------------------------------------------------------------- | -------------- | ----------------- |
| decode (PyAV, single-thread)                                                                   | no             | no                |
| `.cpu().numpy()` roundtrip                                                                     | chunk-level    | pulls **off** GPU |
| `[:, :, ::-1]` BGR copy                                                                        | no             | no                |
| **Ultralytics preprocessing** — letterbox-resize each full-res frame → 640, HWC→CHW, normalize | CPU, per-image | no                |
| **NMS postprocess**                                                                            | CPU, per-image | no                |
| GPU forward                                                                                    | ✅ yes          | ✅ yes             |

**The proof it didn't help is `sm 0` (§4).** If the batched forward were the cost, GPU
compute would spike during detect; instead it stays at zero — the GPU is *waiting*
between batches while the CPU decodes, copies, letterboxes, and runs NMS. Batching a
stage the GPU finishes in milliseconds cannot move a wall that lives on the CPU.

Consequence for the fix order: **only striding cuts the per-frame CPU work** (fewer
decodes, letterboxes, NMS calls); **downscale** shrinks each of those operations; and
**feeding `model.predict` an already-resized CUDA tensor** makes Ultralytics skip the
CPU letterbox/preprocess entirely and keeps the frame on-device. Increasing
`--yoloBatchSize` further does essentially nothing.

### 3. The `asd` stage is mostly a second decode

From the logs, the 39.7 s asd stage is **~30 s decode+crop + ~9 s model forward.**
The video is decoded a **second full time** here just to crop faces. The model
itself — the supposed villain — is 9 seconds.

### 4. GPU is idle during the bottleneck (the headline)

On the 102-min run, during the detect stage, `nvidia-smi dmon -s u`:

```
# gpu   sm   mem   enc   dec   jpg   ofa
    0    0     0     0     0     0     0
    0    0     0     0     0     0     0
```

- **`sm 0`** — compute idle. YOLO barely runs; the stage is stalled on the CPU.
- **`dec 0`** — the hardware decoder (NVDEC) is **unused**. The "GPU decode" path in
  `utils/gpu_video.py` is effectively fake: it decodes on CPU (PyAV), copies each
  frame to a numpy array, then uploads. Detection then pulls it back to CPU
  (`model/yoloFace.py:49`). The A10 does no decode work at all.

Memory: 3.9 GB of 23 GB used, 0% util. The card is asleep.

### 5. The video is decoded 4–5 times

| Pass | Stage                                     | Decodes every frame?     |
| ---- | ----------------------------------------- | ------------------------ |
| 1    | `extract_video` — ffmpeg re-encode to AVI | Yes (decode + re-encode) |
| 2    | `scene_detect` — PySceneDetect            | Yes (CPU)                |
| 3    | `run_face_detection` — YOLO               | Yes (+ CPU roundtrip)    |
| 4    | `run_asd_inference_gpu` — cropping        | Yes (again)              |
| 5    | `visualization` (unless `--metadataOnly`) | Yes (again)              |

### 6. The model runs 6× more than necessary

`main.py:297` scores over `durationSet = [1, 2, 3, 4, 5, 6]` — six full forward
passes over the video for a multi-scale ensemble. Cutting to `[1, 2]` roughly halves
to thirds the ASD compute for near-identical argmax (which face is speaking).

### 7. Costs that only appear at scale (102-min video)

- **preprocess = 7 m 22 s** (was 20 s on the 7-min clip) — the ffmpeg re-encode
  scales badly and produces a giant intermediate AVI that makes every later decode
  slower.
- **scene = 5 m 08 s**, 369 scenes detected (was 16).
- **Memory risk:** `all_talknet_chunks` accumulates every track's features on CPU
  RAM (`main.py:226`). 369 scenes → many tracks → possible OOM/swap on long video.
  *(Result: did NOT bite at 102-min / 553 tracks — see §8. Still unbounded; cap for multi-hour.)*

### 8. Long-video reality check (102-min, higher-res) — the real baseline

Full run completed. Total **5910.7 s = 98.5 min** to process a 6133 s (102-min) clip →
**1.04× realtime.** This is *slower* than the 7-min fixture's 1.96×. **The short fixture
flattered us; the pipeline degrades on real long content.**

| Stage      |             seconds | % of total |      fps | vs 7-min fps          |
| ---------- | ------------------: | ---------: | -------: | --------------------- |
| **detect** | **4025.8** (67 min) |  **68.1%** | **38.1** | was 71.6 — **halved** |
| asd        |   1185.3 (19.8 min) |      20.1% |      129 | was 272               |
| preprocess |       424.8 (7 min) |       7.2% |        — | ffmpeg re-encode      |
| scene      |     264.7 (4.4 min) |       4.5% |        — | —                     |
| track      |                 0.5 |       0.0% |        — | free                  |
| output     |                 9.6 |       0.2% |        — | metadata only         |

153,330 frames · **369 scenes** (was 16) · **553 face tracks** (was 25).

**Why it got slower per-frame — resolution is a first-class cost the code ignores:**
- `chunk_size` auto-collapsed **126 → 52 frames** (the memory calc scales with H×W),
  meaning this clip is ~2.4× more pixels/frame — almost certainly 1080p vs the fixture's
  ~720p. detect fps halved and the asd decode ballooned accordingly.
- **The YOLO path feeds full-resolution frames.** `facedetScale` (default 0.25) is
  **unused** in the in-memory path — nothing is downscaled before detection. So the
  per-frame decode + `.cpu().numpy()` roundtrip cost scales directly with source
  resolution. **Downscale-on-decode is therefore the single biggest lever for HD input**,
  and it is currently missing entirely.

**Where the two big stages actually went (decomposed from logs):**
- `detect` 4026 s ≈ **~1000 s decode + ~3000 s (50 min) YOLO forward + CPU roundtrip.**
  That's ~19.7 ms/frame for a *nano* model — essentially all Python dispatch + roundtrip
  overhead, not compute.
- `asd` 1185 s ≈ **~1012 s second full decode + ~152 s model** (553 tracks). The model —
  the original prime suspect — is **2.6% of total.**

**Good news — the memory risk did not bite:** 553 tracks combined on CPU without OOM;
asd GPU peak 4.9 / 23 GB. `all_talknet_chunks` held, but it's still unbounded and should
be capped/streamed before multi-hour input.

---

## Root cause

**The `detect` stage is bound by single-threaded CPU video decode plus a per-frame
GPU→CPU→GPU roundtrip, while running YOLO on every one of ~153,000 full-resolution
frames.** The GPU is idle throughout, and cost scales with source resolution because
nothing downscales before detection. Everything else is secondary.

---

## Fix roadmap (ordered by leverage)

Budget to hit 12× on the **102-min HD clip**: 6133 s / 12 = **511 s** (currently 5911 s
— we need to cut ~11.5×).

| #   | Fix                                                                                                                                                                                | Attacks                              | Long-video effect       |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ----------------------- |
| 1   | **Strided detection** — YOLO every ~10th frame; `track_shot()` already interpolates gaps (`track_utils.py:118`), scene cuts anchor re-detection. Already designed (never shipped). | per-frame YOLO dispatch (~3000 s)    | ~3000 s → ~300 s        |
| 2   | **Downscale-on-decode** — decode/resize to detection resolution (`facedetScale` is currently unused in the in-memory path). Biggest lever for HD.                                  | resolution-scaled decode + roundtrip | decode ~1000 s → ~300 s |
| 3   | **True NVDEC + keep frames on GPU** — drop the `.cpu().numpy()`/BGR roundtrip, feed YOLO GPU tensors (Ultralytics accepts them). Lights up `dec`, frees the CPU.                   | decode roundtrip                     | compounds #1/#2         |
| 4   | **Decode once, reuse** — one GPU-resident pass feeds detect + crop + scene. Kills the **second ~1012 s decode** in the asd stage.                                                  | passes 2/4/5                         | asd 1185 s → ~200 s     |
| 5   | **Drop the ffmpeg re-encode** — decode the source directly; normalize fps only if needed.                                                                                          | preprocess + intermediate size       | 425 s → ~0              |
| 6   | **Shrink ASD compute (optional)** — `durationSet [1..6] → [1,2]`; bigger batch (asd peak only 4.9/23 GB).                                                                          | model forward                        | model 152 s → ~60 s     |

Rough post-fix budget (102-min clip):

```
decode once, NVDEC, downscaled          ~250–350 s   (feeds detect + crop + scene)
YOLO strided /10 on GPU tensors         ~200–300 s
model forward                           ~150 s
preprocess re-encode + scene folded in  ~0
                                        ─────────────
total                                   ~500–700 s  →  ~9–12× realtime
```

**This is much tighter than the 7-min projection suggested.** 12× on a single HD video
needs *every* fix landing well. Striding alone (fix #1) takes detect from ~4026 s to
~1300 s → total ~3200 s → ~1.9× — a real 2× win, but nowhere near 12× on its own.
Decode (fixes #2–#4) is now co-equal with striding, not secondary.

---

## Throughput cheat for the customer

Single-video latency is the hard problem. But long-to-short is **embarrassingly
parallel across videos and across scene cuts.** Sharding N videos (or N scenes of one
video) across GPUs/workers can hit the customer's *throughput* SLA immediately, even
before single-video latency reaches 12×. Cheapest path to "an hour done in five
minutes" for a batch.

---

## Feasibility verdict

Confirmed against a real 102-min HD clip, the honest verdict is **more cautious than the
7-min fixture implied:**

- **Single-video 12× is possible but not safe to promise.** The realistic baseline is
  **1.04× realtime**, so 12× means cutting ~11.5×. That requires striding **and**
  downscale-on-decode **and** NVDEC/no-roundtrip **and** decode-once all landing well.
  High confidence on **~4–6×** (striding + decode-once + downscale); 12× is the optimistic
  end of the range and depends on the decode stack coming together cleanly.
- **A bigger GPU buys almost nothing** — the A10 is idle now and would stay idle.
  Resolution and CPU decode, not compute, set the cost.
- **The safe customer promise is throughput, not single-video latency.** Long-to-short is
  embarrassingly parallel across videos and scene-cuts (§ below). Sharding hits an
  "hour-in-five-minutes" *aggregate* SLA immediately, and de-risks the timeline while the
  single-video path is rebuilt.

**Recommended first build:** strided detection (fix #1) — highest leverage, already
designed, testable without touching the fragile decode path — followed immediately by
downscale-on-decode (fix #2), the biggest HD lever and currently missing.

---

## Appendix — reproduce

```bash
# one run, per-stage breakdown
uv run python main.py --videoName sample --videoFolder tests/fixtures \
    --asdModel lightasd --metadataOnly
uv run python -m json.tool tests/fixtures/sample/pywork/timings.json

# prove GPU is idle during detect (run in a second terminal while the above runs)
nvidia-smi dmon -s u -d 5      # watch sm and dec columns — both stay 0
```

Related: `docs/runbooks/speed-test-runbook.md` (benchmark how-to).
