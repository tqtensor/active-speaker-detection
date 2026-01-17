# Active Speaker Detection

A full pipeline to detect and highlight active speakers in videos using YOLO for face detection and TalkNet for speaker detection.

---

## Clone the Repository

```bash
git clone https://github.com/MjdMahasneh/active-speaker-detection.git
cd active-speaker-detection
```

---

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
uv sync
```

This will create a virtual environment and install all dependencies including PyTorch with CUDA support.

---

## Run

To run the pipeline, modify configurations in `./config/args.py` and then run `main.py`. Alternatively, you can run the script directly with command line arguments.

```bash
uv run python main.py --videoName video --videoFolder workdir
```

For better performance with GPU, increase batch size and data loader threads:

```bash
uv run python main.py --videoName video --videoFolder workdir --yoloBatchSize 64 --nDataLoaderThread 16
```

For faster processing without video visualization (metadata only):

```bash
uv run python main.py --videoName video --videoFolder workdir --metadataOnly
```

To use a larger YOLO model for better accuracy:

```bash
uv run python main.py --videoName video --videoFolder workdir --yoloVariant m
```

**Note:** video can be in `.mp4` or `.avi` formats.

---

## Output Structure

```
workdir/
└── video/
    ├── pyavi/                 # extracted audio + output video
    ├── pyframes/              # all video frames
    ├── pycrop/                # cropped face clips
    └── pywork/
        ├── tracks.pckl        # face tracks
        ├── scores.pckl        # speaking scores
        ├── speaker_summary.json    # summary of speaker activity
        └── frame_metadata.json     # frame-centric metadata
```

---

## Models

- **YOLOv11-Face**: Face detection (variants: n/s/m/l/x for speed vs accuracy tradeoff)
- **TalkNet**: Audio-visual active speaker detection

---

## Command Line Options

| Option                  | Default   | Description                                                               |
| ----------------------- | --------- | ------------------------------------------------------------------------- |
| `--videoName`           | `video`   | Input video name (without extension)                                      |
| `--videoFolder`         | `workdir` | Path for inputs and outputs                                               |
| `--yoloVariant`         | `n`       | YOLO variant: n (nano), s (small), m (medium), l (large), x (extra-large) |
| `--yoloBatchSize`       | `32`      | Batch size for face detection (increase for better GPU utilization)       |
| `--nDataLoaderThread`   | `10`      | Number of parallel workers for video cropping                             |
| `--speakerThresh`       | `0.6`     | Speaker detection confidence threshold                                    |
| `--minSpeechLen`        | `0.25`    | Minimum speech duration (seconds) to count as speaking                    |
| `--ignoreMultiSpeakers` | `False`   | Skip frames with multiple speakers in visualization                       |
| `--metadataOnly`        | `False`   | Skip video visualization, only produce JSON metadata                      |

---

## Components

- Scene detection via `PySceneDetect`
- Face detection via YOLO (batched inference)
- Face tracking via IOU + interpolation
- Speech classification via TalkNet
- Visualization with speaking durations

---

## Optimizations

- **Batch face detection** (`--yoloBatchSize`): Process multiple frames in a single GPU batch for better utilization.
- **Parallel video cropping** (`--nDataLoaderThread`): Concurrent processing of face crops using ProcessPoolExecutor.
- **YOLO variant selection** (`--yoloVariant`): Choose between speed (nano) and accuracy (extra-large) based on your needs.
- **Metadata-only mode** (`--metadataOnly`): Skip expensive visualization when only JSON output is needed.
- **Frame-centric metadata export**: Comprehensive `frame_metadata.json` with per-frame face data, bounding boxes, and speaking scores.
- **Smart interpolation**: Skipped when face detections have no frame gaps, improving efficiency for continuous tracks.
- **Weighted averaging**: Applied across multi-duration inputs instead of repeating inference.
- **Speaker track isolation**: `get_speaker_track_indices()` identifies actual speaker tracks based on configurable thresholds.

---

## Acknowledgements

This project builds on the great work from:

- [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) for active speaker detection.
- [YOLO-Face](https://github.com/akanametov/yolo-face) for face detection.
