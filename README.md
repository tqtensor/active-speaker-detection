# Active Speaker Detection 🎤👀

A full pipeline to detect and highlight active speakers in videos using YOLO for face detection and TalkNet for speaker detection.

---

## 📁 Clone the Repository

```bash
git clone https://github.com/MjdMahasneh/active-speaker-detection.git
cd active-speaker-detection
```

---

## ⚙️ Setup

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

## ▶️ Run

To run the pipeline, modify configurations in `./config/args.py` and then run `main.py`. Alternatively, you can run the script directly with command line arguments.

```bash
uv run python main.py --videoName video --videoFolder workdir
```

For better performance with GPU, increase batch size and data loader threads:

```bash
uv run python main.py --videoName video --videoFolder workdir --yoloBatchSize 64 --nDataLoaderThread 16
```

**Note:** video can be in `.mp4` or `.avi` formats.

---

## 🗂️ Output Structure

```
workdir/
└── video/
    ├── pyavi/                # extracted audio + output video
    ├── pyframes/             # all video frames
    ├── pycrop/               # cropped face clips
    ├── pywork/               # pickle files and internals
    └── speaker_summary.json  # summary of speaker activity
```

---

## 🧠 Models

- **YOLOv11n-Face**: face detection
- **TalkNet**: audio-visual active speaker detection

---

## 🛠️ Components

- Scene detection via `PySceneDetect`
- Face detection via YOLO
- Face tracking via IOU + interpolation
- Speech classification via TalkNet
- Visualization with speaking durations

---

## 🔧 Improvements Made

- Added batch face detection (`--yoloBatchSize`) for better GPU utilization.
- Added parallel video cropping (`--nDataLoaderThread`) for faster preprocessing.
- Added `--minSpeechLen` to filter out short/non-speech segments.
- Added `--ignoreMultiSpeakers` to ignore segments with multiple speakers.
- Skipped interpolation when face detections have no frame gaps, improving efficiency for continuous tracks.
- Applied weighted averaging across multi-duration inputs instead of repeating inference.
- Added `get_speaker_track_indices()` to isolate actual speaker tracks.
---


## Acknowledgements

This project builds on the great work from:

- [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) for active speaker detection.
- [YOLO-Face](https://github.com/akanametov/yolo-face) for face detection.
