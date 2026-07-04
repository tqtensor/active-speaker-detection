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
    └── pywork/
        ├── tracks.pckl        # face tracks
        ├── scores.pckl        # speaking scores
        ├── speaker_summary.json    # summary of speaker activity
        └── frame_metadata.json     # frame-centric metadata
```

Video frames and face crops are decoded and processed entirely in memory
(GPU-accelerated) and are never written to disk.

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
| `--talknetBatchSize`    | `16`      | Batch size for GPU-batched TalkNet/Light-ASD inference                    |

---

## Components

- Scene detection via `PySceneDetect`
- Face detection via YOLO (batched inference)
- Face tracking via IOU + interpolation
- Speech classification via TalkNet
- Visualization with speaking durations

---

## Performance Optimization Guide

### Quick Start for Best Performance

For maximum speed on a GPU with 24GB+ VRAM (e.g., L4, A100):

```bash
uv run python main.py --videoName video --videoFolder workdir \
    --yoloBatchSize 128 \
    --talknetBatchSize 32 \
    --metadataOnly
```

### Key Optimizations

#### 1. In-Memory GPU Pipeline
Frames and face crops are decoded, cropped, and converted to model features
directly on the GPU in chunks — there is no disk I/O for frames/crops and no
separate "batched" mode to opt into; batched inference is always used.

#### 2. Increase YOLO Batch Size (`--yoloBatchSize`)
Higher batch sizes improve GPU utilization for face detection:
- **16GB VRAM**: `--yoloBatchSize 64`
- **24GB VRAM**: `--yoloBatchSize 128`
- **40GB+ VRAM**: `--yoloBatchSize 200`

#### 3. TalkNet/Light-ASD Batch Size (`--talknetBatchSize`)
Controls how many audio-visual segments are batched per inference call.
Increase for more GPU utilization, decrease if you hit out-of-memory errors.

#### 4. Metadata-Only Mode (`--metadataOnly`)
Skip expensive video visualization when only JSON output is needed:

```bash
uv run python main.py --videoName video --videoFolder workdir --metadataOnly
```

#### 5. YOLO Variant Selection (`--yoloVariant`)
Choose between speed and accuracy:
- `n` (nano): Fastest, good for clear faces
- `s` (small): Balanced
- `m` (medium): Better accuracy
- `l` (large): Best accuracy, slower

### Troubleshooting Performance

1. **Out of memory**: Reduce `--yoloBatchSize` and `--talknetBatchSize`
2. **CPU bottleneck**: Increase `--nDataLoaderThread` (up to CPU core count)

---

## Acknowledgements

This project builds on the great work from:

- [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) for active speaker detection.
- [YOLO-Face](https://github.com/akanametov/yolo-face) for face detection.
