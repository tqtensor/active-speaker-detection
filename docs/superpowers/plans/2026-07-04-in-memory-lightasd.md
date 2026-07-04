# In-Memory Light-ASD Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the offline ASD pipeline fast by decoding video in memory (no `pyframes/` JPEGs, no per-track `.wav`) and swapping TalkNet for the lighter Light-ASD model.

**Architecture:** Two in-memory decode passes over the normalized `pyavi/video.avi`: pass 1 feeds YOLO face detection, pass 2 (already present) feeds GPU face cropping. Audio becomes one global MFCC sliced per track. The ASD model is selectable (`talknet` | `lightasd`); Light-ASD is vendored under `model/lightASD/` and takes byte-identical input tensors to TalkNet, differing only by dropping the cross-attention call. Legacy disk-based paths are removed after the model swap lands.

**Tech Stack:** Python 3.12, PyTorch/CUDA, PyAV (`av`), ultralytics YOLO, `python_speech_features`, ffmpeg-python, pytest (added for unit tests).

## Global Constraints

- Python `>=3.12,<3.13`; deps managed with `uv` (`uv run python ...`, `uv sync`).
- CUDA is **required** — the in-memory path uses `decode_video_chunked` / `crop_faces_gpu`. No CPU compute fallback after Phase 3.
- Video is normalized to **25 fps** by `extract_video`; all frame↔time math assumes 25 fps and MFCC at 100 fps (4× video rate).
- Light-ASD is **MIT-licensed** (Junhua-Liao/Light-ASD). Vendor its source verbatim except for import-path fixes; keep its `LICENSE`.
- Default Light-ASD weights: **`pretrain_AVA_CVPR.model`** — `https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/main/weight/pretrain_AVA_CVPR.model` (4.18 MB). Alt: `finetuning_TalkSet.model` (same URL dir).
- Preserve existing output contract: `pywork/{tracks.pckl,scores.pckl,speaker_summary.json,frame_metadata.json}` and (unless `--metadataOnly`) `pyavi/video_out.avi`. `vidTracks` entries stay `{"track": track, "proc_track": {"x","y","s"}}`.
- Regression note: removing the JPEG re-encode step means detections are computed on raw decoded frames instead of `q=2` JPEGs. Expect **near-identical** scores/speaker decisions, not necessarily byte-identical — the gate is "same speaker/frame decisions within a small score tolerance."

## File Structure

| Path                                        | Responsibility                                                                     | Change                       |
| ------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------- |
| `utils/audio_features.py`                   | Global MFCC compute + per-track slice + proc_track smoothing                       | **new**                      |
| `model/yoloFace.py`                         | In-memory decode → YOLO → dets                                                     | rewrite `run_face_detection` |
| `model/lightASD/`                           | Vendored Light-ASD (`Encoder,Classifier,Model,loss`) + `lightASD` wrapper          | **new package**              |
| `main.py`                                   | Orchestration; model-agnostic inference driver; no disk frames                     | modify                       |
| `config/args.py`                            | Add `--asdModel`, `--lightAsdWeights`; drop `--jpegQscale`, `--useBatched`         | modify                       |
| `utils/video_utils.py`                      | Drop `extract_frames`                                                              | modify                       |
| `utils/track_utils.py`                      | Drop `crop_video`, `extract_audio_only`                                            | modify                       |
| `utils/inference_utils.py`                  | Drop `evaluate_network*`; keep `get_speaker_track_indices`                         | modify                       |
| `utils/helpers.py`                          | `visualization`/`export_metadata` read decoded frames / video info, not `pyframes` | modify                       |
| `utils/dataset.py`, `utils/video_loader.py` | dead after cleanup                                                                 | **delete**                   |
| `tests/`                                    | pytest unit tests + fixtures                                                       | **new**                      |
| `pyproject.toml`                            | add `pytest` to dev extras                                                         | modify                       |

Task order: **1–4 = Phase 1** (kill disk I/O, TalkNet still active, regression-gated). **5–7 = Phase 2** (vendor + swap). **8–9 = Phase 3** (viz/metadata rework, delete legacy). **10 = final integration**.

---

### Task 1: Global MFCC + proc_track helpers

**Files:**
- Create: `utils/audio_features.py`
- Create: `tests/__init__.py` (empty), `tests/test_audio_features.py`
- Modify: `pyproject.toml` (add pytest to `[project.optional-dependencies].dev`)

**Interfaces:**
- Produces:
  - `compute_global_mfcc(audio_path: str) -> numpy.ndarray` — shape `(N, 13)`, MFCC of the whole audio at 16 kHz, `winlen=0.025`, `winstep=0.010`, `numcep=13`.
  - `slice_track_mfcc(global_mfcc: numpy.ndarray, frame_start: int, frame_end: int) -> numpy.ndarray` — rows `[frame_start*4 : (frame_end+1)*4]`.
  - `compute_proc_track(track: dict) -> dict` — `{"x": ndarray, "y": ndarray, "s": ndarray}`, each `scipy.signal.medfilt(..., kernel_size=13)` of per-frame bbox center/half-size, matching the smoothing in the old `crop_video`/`extract_audio_only`.

- [ ] **Step 1: Add pytest to dev deps**

In `pyproject.toml`, under `[project.optional-dependencies]` `dev = [...]`, add:

```toml
  "pytest>=8.0.0",
```

Then run `uv sync --extra dev`.

- [ ] **Step 2: Write failing tests**

`tests/test_audio_features.py`:

```python
import numpy as np
import scipy.io.wavfile as wavfile

from utils.audio_features import (
    compute_global_mfcc,
    slice_track_mfcc,
    compute_proc_track,
)


def test_compute_global_mfcc_shape(tmp_path):
    # 1 second of 16 kHz noise
    sr = 16000
    audio = (np.random.randn(sr) * 1000).astype(np.int16)
    wav = tmp_path / "a.wav"
    wavfile.write(wav, sr, audio)

    mfcc = compute_global_mfcc(str(wav))
    assert mfcc.ndim == 2 and mfcc.shape[1] == 13
    # ~100 frames per second
    assert 95 <= mfcc.shape[0] <= 105


def test_slice_track_mfcc_range():
    mfcc = np.arange(400 * 13).reshape(400, 13).astype(float)
    # frames 10..19 inclusive -> rows [40:80]
    out = slice_track_mfcc(mfcc, 10, 19)
    assert out.shape[0] == 40
    assert np.array_equal(out[0], mfcc[40])
    assert np.array_equal(out[-1], mfcc[79])


def test_compute_proc_track_keys_and_length():
    track = {
        "frame": np.arange(20),
        "bbox": np.tile(np.array([10.0, 20.0, 50.0, 80.0]), (20, 1)),
    }
    proc = compute_proc_track(track)
    assert set(proc) == {"x", "y", "s"}
    assert len(proc["x"]) == 20
    # center x = (10+50)/2 = 30, half-size = max(60,40)/2 = 30
    assert abs(proc["x"][10] - 30.0) < 1e-6
    assert abs(proc["s"][10] - 30.0) < 1e-6
```

- [ ] **Step 3: Run tests, verify they fail**

Run: `uv run pytest tests/test_audio_features.py -v`
Expected: FAIL — `ModuleNotFoundError: utils.audio_features`.

- [ ] **Step 4: Implement `utils/audio_features.py`**

```python
import numpy
import python_speech_features
from scipy import signal
from scipy.io import wavfile


def compute_global_mfcc(audio_path):
    """Computes MFCC features for an entire audio file.

    Args:
        audio_path: Path to a 16 kHz mono WAV file.

    Returns:
        Array of shape (N, 13) with MFCC frames at 100 fps.
    """
    sr, audio = wavfile.read(audio_path)
    return python_speech_features.mfcc(
        audio, sr, numcep=13, winlen=0.025, winstep=0.010
    )


def slice_track_mfcc(global_mfcc, frame_start, frame_end):
    """Slices the global MFCC to a track's inclusive video-frame range.

    Video runs at 25 fps and MFCC at 100 fps, so each video frame maps to
    four MFCC rows.

    Args:
        global_mfcc: Full-clip MFCC array of shape (N, 13).
        frame_start: First video frame index of the track (inclusive).
        frame_end: Last video frame index of the track (inclusive).

    Returns:
        MFCC slice of shape (M, 13).
    """
    return global_mfcc[frame_start * 4 : (frame_end + 1) * 4, :]


def compute_proc_track(track):
    """Computes median-smoothed face center and size for a track.

    Args:
        track: Dict with 'bbox' array of [x1, y1, x2, y2] rows.

    Returns:
        Dict with 'x', 'y', 's' arrays (smoothed center x/y and half-size).
    """
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    return dets
```

- [ ] **Step 5: Run tests, verify pass**

Run: `uv run pytest tests/test_audio_features.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add utils/audio_features.py tests/ pyproject.toml uv.lock
git commit -m "feat: add global MFCC and proc_track helpers"
```

---

### Task 2: In-memory face detection (drop pyframes)

**Files:**
- Modify: `model/yoloFace.py` (rewrite `run_face_detection`)
- Modify: `utils/video_utils.py` (remove `extract_frames`)
- Modify: `main.py` (remove `extract_frames` call + `pyframesPath` creation)
- Create: `tests/test_face_detection.py`
- Create: `tests/fixtures/README.md` (documents the sample clip requirement)

**Interfaces:**
- Consumes: `utils.gpu_video.decode_video_chunked(video_path, chunk_size, device_id) -> yields (chunk_idx, start_frame, frames_tensor[N,H,W,3 uint8 GPU])`; `utils.gpu_video.get_video_info(path) -> {num_frames,fps,width,height}`.
- Produces: `run_face_detection(args, batch_size=32) -> list[list[dict]]` where `dets[frame_idx]` is a list of `{"frame": int, "bbox": [x1,y1,x2,y2], "conf": float}`. Semantics unchanged from the current JPEG-based version; only the frame source changes.

- [ ] **Step 1: Write failing test** (requires a real sample clip)

`tests/fixtures/README.md`:

```markdown
Place a short (2-5 s) real video with at least one visible face + speech at
`tests/fixtures/sample.mp4`. It is git-ignored; integration tests skip if absent.
```

`tests/test_face_detection.py`:

```python
import os
import types

import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.mp4")
pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE), reason="tests/fixtures/sample.mp4 missing"
)


def _prep_video(tmp_path):
    """Runs extract_video/extract_audio into tmp_path, returns args stub."""
    import shutil
    from utils.video_utils import extract_video, extract_audio

    os.makedirs(tmp_path / "pyavi", exist_ok=True)
    args = types.SimpleNamespace(
        videoPath=FIXTURE,
        pyaviPath=str(tmp_path / "pyavi"),
        duration=0,
        start=0,
        nDataLoaderThread=2,
        yoloFaceWeights="./weights/yolo/yolov11n-face.pt",
    )
    extract_video(args)
    extract_audio(args)
    return args


def test_run_face_detection_returns_per_frame_dets(tmp_path):
    from model.yoloFace import run_face_detection
    from utils.gpu_video import get_video_info

    args = _prep_video(tmp_path)
    dets = run_face_detection(args, batch_size=8)

    n = get_video_info(args.videoFilePath)["num_frames"]
    assert len(dets) == n
    # every entry is a list; at least one frame has a detection
    assert all(isinstance(d, list) for d in dets)
    assert any(len(d) > 0 for d in dets)
    for d in dets:
        for face in d:
            assert set(face) == {"frame", "bbox", "conf"}
            assert len(face["bbox"]) == 4
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_face_detection.py -v`
Expected: FAIL — current `run_face_detection` globs `args.pyframesPath` which the stub does not set (`AttributeError`), or skip if no fixture. (If skipped, add the fixture before proceeding.)

- [ ] **Step 3: Rewrite `run_face_detection` in `model/yoloFace.py`**

Replace the whole file body below the imports. New imports at top: `import numpy`, `from ultralytics import YOLO`, `from utils.gpu_video import decode_video_chunked, get_video_info`, `from config.logging_config import get_logger`. Remove `glob`, `cv2`, `sys` if now unused (keep `os`, `pickle`).

```python
import os
import pickle

import numpy
from ultralytics import YOLO

from config.logging_config import get_logger
from utils.gpu_video import decode_video_chunked, get_video_info

logger = get_logger(__name__)


def load_yolo_model(weights_path):
    """Loads a YOLO model from the specified weights file."""
    return YOLO(weights_path)


def run_face_detection(args, batch_size=32):
    """Runs batched face detection over in-memory decoded frames.

    Decodes ``args.videoFilePath`` in chunks (no intermediate JPEGs), runs
    YOLO in batches, and returns per-frame detections. Detections are computed
    on BGR frames to match the previous cv2-based behavior.

    Args:
        args: Config with videoFilePath and yoloFaceWeights.
        batch_size: Frames per YOLO forward pass.

    Returns:
        List indexed by frame; each element is a list of
        {"frame", "bbox", "conf"} dicts.
    """
    model = YOLO(args.yoloFaceWeights)

    num_frames = get_video_info(args.videoFilePath)["num_frames"]
    dets = [None] * num_frames

    for _chunk_idx, start_frame, frames in decode_video_chunked(
        args.videoFilePath, chunk_size=max(batch_size, 256), device_id=0
    ):
        # (N,H,W,3) uint8 RGB on GPU -> CPU numpy, RGB->BGR to match cv2.imread
        chunk = frames.cpu().numpy()[:, :, :, ::-1]
        del frames

        for b in range(0, len(chunk), batch_size):
            batch_imgs = [chunk[i] for i in range(b, min(b + batch_size, len(chunk)))]
            results = model.predict(batch_imgs, conf=0.7, iou=0.5, verbose=False)
            for i, result in enumerate(results):
                fidx = start_frame + b + i
                dets[fidx] = [
                    {
                        "frame": fidx,
                        "bbox": box.xyxy.cpu().numpy().tolist()[0],
                        "conf": float(box.conf.item()),
                    }
                    for box in result.boxes
                ]

    # Any frame the decoder skipped stays an empty detection list.
    dets = [d if d is not None else [] for d in dets]

    with open(os.path.join(args.pyworkPath, "faces.pckl"), "wb") as fil:
        pickle.dump(dets, fil)

    return dets
```

Note: `model.predict` accepts a list of HWC uint8 numpy arrays (BGR), same convention as the prior `cv2.imread` inputs.

- [ ] **Step 4: Remove `extract_frames` from `utils/video_utils.py`**

Delete the entire `extract_frames` function (lines defining it). Leave `extract_video`, `extract_audio`, `extract_MFCC` intact.

- [ ] **Step 5: Update `main.py`**

Remove the import `extract_frames` from `from utils.video_utils import extract_audio, extract_frames, extract_video` → `from utils.video_utils import extract_audio, extract_video`.

In `main()`, delete the line `extract_frames(args)`.

In `prepare_paths(args)`: remove `args.pyframesPath = os.path.join(args.savePath, "pyframes")` and `os.makedirs(args.pyframesPath)`. (Leave `pyworkPath`, `pyaviPath`, `pycropPath` — `pycropPath` still holds nothing after Task 3 but is harmless; it is removed in Task 9.)

- [ ] **Step 6: Run test, verify pass**

Run: `uv run pytest tests/test_face_detection.py -v`
Expected: PASS (1 passed) — requires `tests/fixtures/sample.mp4` and YOLO weights present (`uv run python main.py ...` once, or let `download_weights` fetch them in Task 10; for this test, ensure `./weights/yolo/yolov11n-face.pt` exists).

- [ ] **Step 7: Commit**

```bash
git add model/yoloFace.py utils/video_utils.py main.py tests/
git commit -m "perf: run face detection on in-memory frames, drop pyframes JPEGs"
```

---

### Task 3: Global-MFCC audio in GPU inference (drop per-track wav)

**Files:**
- Modify: `main.py` (`run_talknet_inference_gpu`: use global MFCC + `compute_proc_track`; remove `extract_audio_worker` audio stage)

**Interfaces:**
- Consumes: `utils.audio_features.compute_global_mfcc`, `slice_track_mfcc`, `compute_proc_track` (Task 1).
- Produces: `run_talknet_inference_gpu(allTracks, args) -> (vidTracks, scores)` — same shapes as before; `vidTracks[i] = {"track": track, "proc_track": {...}}` now built in-memory.

- [ ] **Step 1: Replace the audio + feature-prep section of `run_talknet_inference_gpu`**

In `main.py`, inside `run_talknet_inference_gpu`, delete:
- the `from scipy.io import wavfile` import and `import python_speech_features` (move to top-level or replace by helper import),
- the entire "Step 1: Extract audio for all tracks in parallel" `ProcessPoolExecutor` block that builds `vidTracks` via `extract_audio_worker`.

Add near the top of the function:

```python
from utils.audio_features import (
    compute_global_mfcc,
    slice_track_mfcc,
    compute_proc_track,
)

# One MFCC for the whole clip; sliced per track below (no per-track wav I/O).
global_mfcc = compute_global_mfcc(args.audioFilePath)

# proc_track (smoothed centers/size) needed by visualization/metadata.
vidTracks = [
    {"track": track, "proc_track": compute_proc_track(track)}
    for track in allTracks
]
```

- [ ] **Step 2: Replace per-track audio load in the "Prepare TalkNet features" loop**

Find the block:

```python
        videoFeature = talknet_features[track_idx].numpy()  # Already on CPU

        # Load audio features
        _, audio = wavfile.read(crop_path + ".wav")
        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010
        )
```

Replace with:

```python
        videoFeature = talknet_features[track_idx].numpy()  # Already on CPU

        # Slice the global MFCC to this track's frame range (no disk I/O)
        track_frames = numpy.array(allTracks[track_idx]["frame"])
        audioFeature = slice_track_mfcc(
            global_mfcc, int(track_frames[0]), int(track_frames[-1])
        )
```

Also delete the now-unused `crop_path = os.path.join(args.pycropPath, "%05d" % track_idx)` line and any other reference to `crop_path` in this loop.

- [ ] **Step 3: Remove the `extract_audio_worker` function and its import**

In `main.py` delete the `extract_audio_worker` function definition and remove `extract_audio_only` from `from utils.track_utils import crop_video, extract_audio_only, scene_detect, track_shot` → `from utils.track_utils import crop_video, scene_detect, track_shot`. (`crop_video` still imported for the standard path until Task 9.)

- [ ] **Step 4: Verify no remaining per-track wav references**

Run: `grep -n "extract_audio_only\|wavfile\|\.wav\"" main.py`
Expected: no matches for `extract_audio_only`; `wavfile` gone from `run_talknet_inference_gpu`.

- [ ] **Step 5: Smoke-run the GPU inference path** (needs fixture + weights)

Run: `uv run python main.py --videoName sample --videoFolder tests/fixtures --metadataOnly`
(First copy/symlink `tests/fixtures/sample.mp4`; TalkNet still the model here.)
Expected: completes; `tests/fixtures/sample/pyframes` does NOT exist; no `*.wav` under `pycrop/`; `pywork/frame_metadata.json` written.

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "perf: slice global MFCC per track, drop per-track wav extraction"
```

---

### Task 4: Phase-1 regression gate

**Files:**
- Create: `tests/test_phase1_regression.py`

**Interfaces:**
- Consumes: full pipeline via `main.py` (TalkNet model).

- [ ] **Step 1: Capture a baseline BEFORE Phase 1 changes**

If not already captured, check out `master`, run the pipeline on the fixture with `--metadataOnly`, and copy `pywork/frame_metadata.json` to `tests/fixtures/baseline_talknet_metadata.json`. Return to the feature branch. (If a baseline cannot be produced, document the exact command in the test file and mark it `xfail` with a reason — do not fabricate numbers.)

- [ ] **Step 2: Write the regression test**

`tests/test_phase1_regression.py`:

```python
import json
import os
import subprocess

import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
BASELINE = os.path.join(FIX, "baseline_talknet_metadata.json")
pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(FIX, "sample.mp4")) and os.path.exists(BASELINE)),
    reason="fixture clip or baseline metadata missing",
)


def _speaking_decisions(meta):
    out = {}
    for fr in meta["frames"]:
        for face in fr["faces"]:
            out[(fr["frame_id"], face["track_id"])] = face["is_speaking"]
    return out


def test_phase1_matches_baseline(tmp_path):
    subprocess.run(
        [
            "uv", "run", "python", "main.py",
            "--videoName", "sample", "--videoFolder", FIX,
            "--asdModel", "talknet", "--metadataOnly",
        ],
        check=True,
    )
    with open(os.path.join(FIX, "sample", "pywork", "frame_metadata.json")) as f:
        new = json.load(f)
    with open(BASELINE) as f:
        base = json.load(f)

    nd, bd = _speaking_decisions(new), _speaking_decisions(base)
    # Same set of (frame, track) speaking decisions >= 99% agreement.
    common = set(nd) & set(bd)
    assert common, "no overlapping (frame, track) keys"
    agree = sum(1 for k in common if nd[k] == bd[k]) / len(common)
    assert agree >= 0.99, f"speaking-decision agreement {agree:.3f} < 0.99"
```

Note: this test references `--asdModel talknet`, added in Task 6. Until then, run it without that flag (TalkNet is the only model). Add the flag when Task 6 lands.

- [ ] **Step 3: Run the regression test**

Run: `uv run pytest tests/test_phase1_regression.py -v`
Expected: PASS (agreement ≥ 0.99). If it fails, investigate RGB→BGR ordering (Task 2) and MFCC slice boundaries (Task 1) before proceeding.

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase1_regression.py
git commit -m "test: phase-1 regression gate vs TalkNet baseline"
```

---

### Task 5: Vendor Light-ASD model package

**Files:**
- Create: `model/lightASD/__init__.py`
- Create: `model/lightASD/Encoder.py`, `model/lightASD/Classifier.py`, `model/lightASD/Model.py`, `model/lightASD/loss.py` (verbatim from upstream, import fixes only)
- Create: `model/lightASD/lightASD.py` (inference wrapper)
- Create: `model/lightASD/LICENSE` (upstream MIT)
- Create: `tests/test_lightasd_model.py`

**Interfaces:**
- Produces: `lightASD` class with `.model` (`ASD_Model`), `.lossAV`, `.lossV`, `.loadParameters(path)`, `.eval()`. Forward methods on `.model`: `forward_audio_frontend(a[B,T*4,13]) -> [B,T,128]`, `forward_visual_frontend(v[B,T,112,112]) -> [B,T,128]`, `forward_audio_visual_backend(a,v) -> [B*T,128]`. `lossAV.forward(out, labels=None) -> numpy[B*T]` per-frame scores.

- [ ] **Step 1: Fetch upstream source into the package**

```bash
mkdir -p model/lightASD
base="https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/main"
curl -sL "$base/model/Encoder.py"    -o model/lightASD/Encoder.py
curl -sL "$base/model/Classifier.py" -o model/lightASD/Classifier.py
curl -sL "$base/model/Model.py"      -o model/lightASD/Model.py
curl -sL "$base/loss.py"             -o model/lightASD/loss.py
curl -sL "$base/LICENSE"             -o model/lightASD/LICENSE
```

- [ ] **Step 2: Fix imports in `model/lightASD/Model.py`**

Change:

```python
from model.Classifier import BGRU
from model.Encoder import visual_encoder, audio_encoder
```

to:

```python
from .Classifier import BGRU
from .Encoder import visual_encoder, audio_encoder
```

(`Encoder.py`, `Classifier.py`, `loss.py` have no cross-package imports and need no edits.)

- [ ] **Step 3: Create the wrapper `model/lightASD/lightASD.py`**

```python
import torch
import torch.nn as nn

from config.logging_config import get_logger

from .loss import lossAV, lossV
from .Model import ASD_Model

logger = get_logger(__name__)


class lightASD(nn.Module):
    """Inference wrapper for the Light-ASD model.

    Mirrors the interface used at the pipeline's inference call site: builds
    the model and loss heads on CUDA, exposes ``loadParameters`` for the
    upstream checkpoint format, and delegates forward passes to ``self.model``.
    """

    def __init__(self, **kwargs):
        super(lightASD, self).__init__()
        self.model = ASD_Model().cuda()
        self.lossAV = lossAV().cuda()
        self.lossV = lossV().cuda()
        n = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"Light-ASD para number = {n:.2f}M")

    def loadParameters(self, path):
        """Loads upstream Light-ASD weights, tolerating a 'module.' prefix.

        Args:
            path: Path to a Light-ASD .model checkpoint.
        """
        selfState = self.state_dict()
        loadedState = torch.load(path, map_location="cpu")
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    logger.warning(f"{origName} is not in the model.")
                    continue
            if selfState[name].size() != loadedState[origName].size():
                logger.warning(
                    f"Wrong parameter length: {origName}, model: "
                    f"{selfState[name].size()}, loaded: {loadedState[origName].size()}"
                )
                continue
            selfState[name].copy_(param)
```

- [ ] **Step 4: Create `model/lightASD/__init__.py`**

```python
from .lightASD import lightASD

__all__ = ["lightASD"]
```

- [ ] **Step 5: Write the model test**

`tests/test_lightasd_model.py`:

```python
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_lightasd_forward_shapes():
    from model.lightASD import lightASD

    net = lightASD()
    net.eval()
    T = 25
    v = torch.rand(1, T, 112, 112).cuda() * 255  # raw 0-255 grayscale
    a = torch.rand(1, T * 4, 13).cuda()

    with torch.no_grad():
        ea = net.model.forward_audio_frontend(a)
        ev = net.model.forward_visual_frontend(v)
        out = net.model.forward_audio_visual_backend(ea, ev)
        score = net.lossAV.forward(out, labels=None)

    score = np.asarray(score).reshape(-1)
    assert score.shape[0] == T
    assert np.isfinite(score).all()
```

- [ ] **Step 6: Run the test**

Run: `uv run pytest tests/test_lightasd_model.py -v`
Expected: PASS — confirms input shapes match TalkNet's and the frontend→backend→loss chain runs.

- [ ] **Step 7: Commit**

```bash
git add model/lightASD/ tests/test_lightasd_model.py
git commit -m "feat: vendor Light-ASD model (MIT) with inference wrapper"
```

---

### Task 6: Light-ASD weight download + model-selection args

**Files:**
- Modify: `config/args.py` (add `--asdModel`, `--lightAsdWeights`)
- Modify: `main.py` (`download_weights`: fetch Light-ASD weights when selected)

**Interfaces:**
- Produces: `args.asdModel ∈ {"talknet","lightasd"}` (default `"lightasd"`), `args.lightAsdWeights` (default `./weights/lightasd/pretrain_AVA_CVPR.model`).

- [ ] **Step 1: Add args in `config/args.py`**

After the `--talkNetWeights` argument, add:

```python
    parser.add_argument(
        "--asdModel",
        type=str,
        default="lightasd",
        choices=["talknet", "lightasd"],
        help="Active speaker model: lightasd (default, fast) or talknet",
    )
    parser.add_argument(
        "--lightAsdWeights",
        type=str,
        default="./weights/lightasd/pretrain_AVA_CVPR.model",
        help="Path for the pretrained Light-ASD model",
    )
```

- [ ] **Step 2: Add the Light-ASD constant + download in `main.py`**

Near `YOLO_FACE_URLS`, add:

```python
LIGHTASD_WEIGHT_URL = (
    "https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/main/"
    "weight/pretrain_AVA_CVPR.model"
)
```

In `download_weights(args)`, gate the TalkNet download on selection and add the Light-ASD download. Wrap the existing TalkNet block with `if args.asdModel == "talknet":`, then append:

```python
    if args.asdModel == "lightasd" and not os.path.isfile(args.lightAsdWeights):
        os.makedirs(os.path.dirname(args.lightAsdWeights), exist_ok=True)
        logger.info(f"Downloading Light-ASD weights from {LIGHTASD_WEIGHT_URL}...")
        result = subprocess.run(
            ["wget", "-q", "--show-progress", "-L", LIGHTASD_WEIGHT_URL,
             "-O", args.lightAsdWeights],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download Light-ASD weights: {result.stderr}")
        torch.load(args.lightAsdWeights, map_location="cpu")  # validate
```

- [ ] **Step 3: Verify download works**

Run: `uv run python -c "from config.args import get_args" && rm -f ./weights/lightasd/pretrain_AVA_CVPR.model && uv run python - <<'PY'
import types, main
a = types.SimpleNamespace(asdModel="lightasd",
    lightAsdWeights="./weights/lightasd/pretrain_AVA_CVPR.model",
    talkNetWeights="./weights/talknet/pretrain_TalkSet.model",
    yoloFaceWeights="./weights/yolo/yolov11n-face.pt", yoloVariant="n")
main.download_weights(a)
import os; print("OK", os.path.getsize(a.lightAsdWeights))
PY`
Expected: prints `OK 4175289` (or close).

- [ ] **Step 4: Commit**

```bash
git add config/args.py main.py
git commit -m "feat: add asdModel selection and Light-ASD weight download"
```

---

### Task 7: Model-agnostic inference driver + swap at call site

**Files:**
- Modify: `main.py` (rename `run_talknet_inference_gpu` → `run_asd_inference_gpu`; build model by `args.asdModel`; drop `forward_cross_attention` for Light-ASD; remove GPU→standard fallback in `main()`)

**Interfaces:**
- Consumes: `model.talkNet.talkNet`, `model.lightASD.lightASD`.
- Produces: `run_asd_inference_gpu(allTracks, args) -> (vidTracks, scores)`.

- [ ] **Step 1: Build the model by selection**

In the renamed `run_asd_inference_gpu`, replace:

```python
    # Load TalkNet model
    s = talkNet()
    s.loadParameters(args.talkNetWeights)
    s.eval()
```

with:

```python
    # Load the selected ASD model
    if args.asdModel == "lightasd":
        from model.lightASD import lightASD
        s = lightASD()
        s.loadParameters(args.lightAsdWeights)
        use_cross_attention = False
    else:
        s = talkNet()
        s.loadParameters(args.talkNetWeights)
        use_cross_attention = True
    s.eval()
```

- [ ] **Step 2: Guard the cross-attention call in the batched forward**

In the inference loop, replace:

```python
                embedA = s.model.forward_audio_frontend(inputA)
                embedV = s.model.forward_visual_frontend(inputV)
                embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                out = s.model.forward_audio_visual_backend(embedA, embedV)
                scores = s.lossAV.forward(out, labels=None)
```

with:

```python
                embedA = s.model.forward_audio_frontend(inputA)
                embedV = s.model.forward_visual_frontend(inputV)
                if use_cross_attention:
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                out = s.model.forward_audio_visual_backend(embedA, embedV)
                scores = s.lossAV.forward(out, labels=None)
```

- [ ] **Step 3: Rename the function and its call; drop the standard fallback**

Rename `def run_talknet_inference_gpu` → `def run_asd_inference_gpu`. In `main()` replace the try/except:

```python
    try:
        vidTracks, scores = run_talknet_inference_gpu(allTracks, args)
    except Exception as e:
        logger.error(f"GPU mode failed ({e}), falling back to standard mode...")
        vidTracks, scores = run_talknet_inference_standard(allTracks, args)
```

with:

```python
    vidTracks, scores = run_asd_inference_gpu(allTracks, args)
```

(The standard function is deleted in Task 9.)

- [ ] **Step 4: Handle `lossAV` return shape for Light-ASD**

Light-ASD's `lossAV.forward(out, labels=None)` returns a 1-D numpy array of length `B*T`; the batched loop indexes `scores[i]`. Confirm the per-segment extraction still works: the batched code slices `scores[i][:actual_frames]`. Since Light-ASD returns a flat `(B*T,)` array (not per-sample lists), reshape before indexing. In the segment-store block, replace:

```python
                for i, (track_idx, seg_idx, audio_seg, _) in enumerate(batch):
                    actual_frames = audio_seg.shape[0] // 4
                    seg_score = (
                        scores[i][:actual_frames]
                        if isinstance(scores[i], list)
                        else [scores[i]]
                    )
                    segment_scores[(track_idx, seg_idx)] = seg_score
```

with:

```python
                # Normalize scores to a per-sample list of per-frame arrays.
                scores_np = numpy.asarray(scores).reshape(len(batch), -1)
                for i, (track_idx, seg_idx, audio_seg, _) in enumerate(batch):
                    actual_frames = audio_seg.shape[0] // 4
                    seg_score = scores_np[i][:actual_frames].tolist()
                    segment_scores[(track_idx, seg_idx)] = seg_score
```

This works for both models: TalkNet's `lossAV` with `labels=None` also yields one score per frame per batch sample; `reshape(len(batch), -1)` recovers the `[i]`-th sample's frames. Verify TalkNet regression (Task 4) still passes after this change.

- [ ] **Step 5: Run both models on the fixture**

```bash
uv run python main.py --videoName sample --videoFolder tests/fixtures --asdModel talknet  --metadataOnly
uv run python main.py --videoName sample --videoFolder tests/fixtures --asdModel lightasd --metadataOnly
uv run pytest tests/test_phase1_regression.py -v
```
Expected: both runs complete; regression test (TalkNet) still ≥0.99. Inspect Light-ASD `speaker_summary.json` for plausible speakers.

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat: select ASD model at inference, wire Light-ASD forward path"
```

---

### Task 8: Rework visualization + metadata to avoid pyframes

**Files:**
- Modify: `utils/helpers.py` (`visualization`, `export_metadata`)

**Interfaces:**
- Consumes: `utils.gpu_video.get_video_info`; a CPU frame decoder for visualization (add `decode_video_frames_cpu`).

- [ ] **Step 1: Add a simple CPU frame iterator to `utils/gpu_video.py`**

```python
def decode_video_frames_bgr(video_path: str):
    """Yields decoded frames as BGR uint8 numpy arrays (cv2 convention).

    Args:
        video_path: Path to the video file.

    Yields:
        (frame_index, ndarray[H,W,3] uint8 BGR).
    """
    container = av.open(video_path)
    for idx, frame in enumerate(container.decode(video=0)):
        rgb = frame.to_ndarray(format="rgb24")
        yield idx, rgb[:, :, ::-1].copy()
    container.close()
```

- [ ] **Step 2: Rewrite `visualization` frame source in `utils/helpers.py`**

Replace the `flist = sorted(glob.glob(...pyframes...*.jpg))` + `faces = [[] for _ in range(len(flist))]` head with a frame count from video info:

```python
    from utils.gpu_video import get_video_info, decode_video_frames_bgr

    total_frames = get_video_info(args.videoFilePath)["num_frames"]
    faces = [[] for _ in range(total_frames)]
```

Replace `firstImage = cv2.imread(flist[0]); fw, fh = ...` by reading dimensions from video info:

```python
    info = get_video_info(args.videoFilePath)
    fw, fh = info["width"], info["height"]
```

Replace the drawing loop `for fidx, fname in enumerate(flist): image = cv2.imread(fname)` with:

```python
    for fidx, image in tqdm.tqdm(
        decode_video_frames_bgr(args.videoFilePath), total=total_frames
    ):
```

(the loop body is unchanged; it already uses `image`). Remove the now-unused `glob` import if nothing else needs it (keep `cv2` — still used for drawing).

- [ ] **Step 3: Fix `export_metadata` frame count**

Replace:

```python
    flist = sorted(glob.glob(os.path.join(args.pyframesPath, "*.jpg")))
    total_frames = len(flist)
```

with:

```python
    from utils.gpu_video import get_video_info

    total_frames = get_video_info(args.videoFilePath)["num_frames"]
```

- [ ] **Step 4: Run visualization end-to-end** (no `--metadataOnly`)

Run: `uv run python main.py --videoName sample --videoFolder tests/fixtures --asdModel lightasd`
Expected: `tests/fixtures/sample/pyavi/video_out.avi` produced with boxes; no `pyframes/` directory created.

- [ ] **Step 5: Commit**

```bash
git add utils/helpers.py utils/gpu_video.py
git commit -m "perf: decode frames in-memory for visualization and metadata"
```

---

### Task 9: Delete legacy disk path + dead code + args cleanup

**Files:**
- Modify: `main.py` (delete `run_talknet_inference_standard`, `crop_video_worker`; drop unused imports; drop `pycropPath`)
- Modify: `utils/track_utils.py` (delete `crop_video`; `extract_audio_only` already gone in Task 3 if not, delete now)
- Modify: `utils/inference_utils.py` (delete `evaluate_network`, `evaluate_network_batched`; keep `get_speaker_track_indices`)
- Delete: `utils/dataset.py`, `utils/video_loader.py`
- Modify: `config/args.py` (remove `--jpegQscale`, `--useBatched`, `--talknetBatchSize` doc if unused — keep `--talknetBatchSize`, still used by GPU batch loop)
- Modify: `README.md` (update Output Structure — remove `pyframes/`, `pycrop/`)

**Interfaces:** none new; removals only.

- [ ] **Step 1: Delete dead functions in `main.py`**

Remove `run_talknet_inference_standard` and `crop_video_worker` entirely. Remove imports that are now unused: `from utils.track_utils import crop_video, scene_detect, track_shot` → `from utils.track_utils import scene_detect, track_shot`. Remove `glob` if unused. In `prepare_paths`, remove `args.pycropPath` assignment and its `os.makedirs`. Remove any remaining `pycropPath` reference in `run_asd_inference_gpu` (the audio-params/crop-params lists are already gone).

- [ ] **Step 2: Delete `crop_video` and `extract_audio_only` in `utils/track_utils.py`**

Remove both functions. Keep `scene_detect`, `bb_intersection_over_union`, `track_shot`. Remove now-unused imports (`glob`, `cv2`, `ffmpeg` if unused; keep `numpy`, `scipy.signal`, `scipy.interpolate`, `scenedetect`).

- [ ] **Step 3: Delete evaluate functions in `utils/inference_utils.py`**

Remove `evaluate_network` and `evaluate_network_batched`. Keep `get_speaker_track_indices`. Remove now-unused imports (`math`, `tqdm`, `torch`, `talkNet` import).

- [ ] **Step 4: Delete dead modules**

```bash
git rm utils/dataset.py utils/video_loader.py
```

- [ ] **Step 5: Remove obsolete args**

In `config/args.py` delete the `--jpegQscale` and `--useBatched` arguments. (Keep `--talknetBatchSize` — the GPU batch loop still reads `args.talknetBatchSize`.)

- [ ] **Step 6: Verify nothing imports the removed symbols**

Run:
```bash
grep -rn "extract_frames\|crop_video\|extract_audio_only\|evaluate_network\|useBatched\|jpegQscale\|pyframesPath\|pycropPath\|run_talknet_inference_standard\|dataset\|video_loader" \
  main.py utils/ model/ config/
```
Expected: no matches (except comments/README). Fix any stragglers.

- [ ] **Step 7: Update README Output Structure**

In `README.md`, edit the output tree: remove the `pyframes/` and `pycrop/` lines; note frames/audio are processed in memory. Update the `--metadataOnly` / performance sections if they mention JPEG frames.

- [ ] **Step 8: Full run + tests after cleanup**

```bash
uv run pytest -v
uv run python main.py --videoName sample --videoFolder tests/fixtures --asdModel lightasd
```
Expected: tests pass (CUDA-gated ones may skip on non-GPU boxes); pipeline completes; only `pyavi/` and `pywork/` outputs exist.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: remove disk-based legacy path and dead code"
```

---

### Task 10: Final integration + timing report

**Files:**
- Modify: `main.py` (add per-stage timing logs)

**Interfaces:** none.

- [ ] **Step 1: Add coarse stage timing in `main()`**

Wrap the five stages with `time.perf_counter()` deltas logged via `logger.info`, e.g.:

```python
import time
...
    t = time.perf_counter()
    extract_video(args); extract_audio(args)
    logger.info(f"[timing] preprocess {time.perf_counter()-t:.1f}s")

    t = time.perf_counter()
    scene = scene_detect(args)
    faces = run_face_detection(args, batch_size=args.yoloBatchSize)
    logger.info(f"[timing] detect {time.perf_counter()-t:.1f}s")

    t = time.perf_counter()
    vidTracks, scores = run_asd_inference_gpu(allTracks, args)
    logger.info(f"[timing] asd {time.perf_counter()-t:.1f}s")
```

- [ ] **Step 2: Compare against a longer real video**

Run the pipeline on a representative (≥1 min) clip with `--asdModel lightasd` and again with `--asdModel talknet`. Record wall-clock per stage from the `[timing]` logs. Expected: detect + asd stages drop sharply vs. the pre-change `master` run (no JPEG write/read, no per-track wav, lighter model).

- [ ] **Step 3: Sanity-check outputs**

Confirm `pyavi/video_out.avi` marks the correct speaker and `speaker_summary.json` intervals are plausible for both models.

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "chore: add per-stage timing logs"
```

---

## Self-Review

**Spec coverage:**
- Kill `pyframes` JPEGs → Task 2. ✓
- Kill per-track wav → Task 3. ✓
- In-memory viz → Task 8. ✓
- Vendor Light-ASD + weights → Tasks 5–6. ✓
- Model swap / drop cross-attention → Task 7. ✓
- Remove legacy disk path entirely → Task 9. ✓
- CUDA-required assumption → Global Constraints + CUDA-gated tests. ✓
- `--asdModel` / `--lightAsdWeights` config → Task 6. ✓
- Testing (regression, sanity, timing) → Tasks 4, 5, 10. ✓

**Placeholder scan:** No TBD/TODO. Each code step shows concrete code. Fixture requirement (`tests/fixtures/sample.mp4`) and baseline capture are explicit (Task 4 Step 1) rather than fabricated.

**Type consistency:** `run_talknet_inference_gpu` → `run_asd_inference_gpu` renamed consistently (Tasks 7, 9). `vidTracks` entry shape `{"track","proc_track"}` consistent (Task 3 build, Task 8 consumers). `compute_global_mfcc`/`slice_track_mfcc`/`compute_proc_track` signatures consistent across Tasks 1 and 3. `lossAV.forward(...)→numpy` handling unified in Task 7 Step 4 for both models.

**Known risk carried into execution:** Task 7 Step 4's `reshape(len(batch), -1)` assumes both models emit exactly one score per MFCC-derived video frame per batch sample. Verified for Light-ASD by Task 5 test; verified for TalkNet by the Task 4 regression gate re-run in Task 7 Step 5. If the regression drops below 0.99 after Step 4, the reshape assumption is wrong for padded batches — fall back to per-sample slicing using each segment's true frame length before padding.
