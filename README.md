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
| `--useBatched`          | `False`   | Use batched TalkNet inference (2-3x faster)                               |
| `--talknetBatchSize`    | `16`      | Batch size for TalkNet when using `--useBatched`                          |

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
    --useBatched \
    --talknetBatchSize 32 \
    --metadataOnly
```

### Optimization Tiers

| Tier     | Speedup | Command Additions                                 | Best For                |
| -------- | ------- | ------------------------------------------------- | ----------------------- |
| Baseline | 1x      | (none)                                            | Debugging, small videos |
| Fast     | 2-3x    | `--useBatched`                                    | General use             |
| Faster   | 4-6x    | `--useBatched --yoloBatchSize 64`                 | Production              |
| Maximum  | 6-10x   | `--useBatched --yoloBatchSize 128 --metadataOnly` | Batch processing        |

### Key Optimizations

#### 1. Batched TalkNet Inference (`--useBatched`)
Processes multiple face tracks in parallel instead of sequentially. **Recommended for all use cases.**

```bash
uv run python main.py --videoName video --videoFolder workdir --useBatched
```

#### 2. Increase YOLO Batch Size (`--yoloBatchSize`)
Higher batch sizes improve GPU utilization for face detection:
- **16GB VRAM**: `--yoloBatchSize 64`
- **24GB VRAM**: `--yoloBatchSize 128`
- **40GB+ VRAM**: `--yoloBatchSize 200`

#### 3. GPU-Accelerated Video Loading
The pipeline automatically uses `decord` for fast video decoding (2-3x faster than OpenCV). GPU decoding is attempted first with automatic fallback to CPU.

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

### GPU Memory Requirements

| Video Length | Frames @25fps | Est. GPU Memory | Recommended GPU |
| ------------ | ------------- | --------------- | --------------- |
| 5 min        | 7,500         | ~6 GB           | Any modern GPU  |
| 10 min       | 15,000        | ~12 GB          | L4 (24GB) ✓     |
| 30 min       | 45,000        | ~36 GB          | A100 (40GB)     |
| 60 min       | 90,000        | ~72 GB          | Needs chunking  |

### Troubleshooting Performance

1. **GPU utilization low during TalkNet**: Use `--useBatched` flag
2. **Out of memory**: Reduce `--yoloBatchSize` and `--talknetBatchSize`
3. **Slow video loading**: Ensure `decord` is installed (`uv sync`)
4. **CPU bottleneck**: Increase `--nDataLoaderThread` (up to CPU core count)

---

## Acknowledgements

This project builds on the great work from:

- [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) for active speaker detection.
- [YOLO-Face](https://github.com/akanametov/yolo-face) for face detection.
