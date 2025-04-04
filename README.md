# Active Speaker Detection 🎤👀

A full pipeline to detect and highlight active speakers in videos using YOLO for face detection and TalkNet for speaker detection.

---

## 📁 Clone the Repository

```bash
git clone https://github.com/your-username/active-speaker-detection.git
cd active-speaker-detection
```

---

## ⚙️ Setup

### 1. Create Conda Environment

```bash
conda create --name active_speaker python=3.9 -y
conda activate active_speaker
```

### 2. Install Dependencies
💡 For GPU support, install PyTorch matching your CUDA version. Check [pytorch.org](https://pytorch.org/get-started/locally/) for installation instructions.

Example for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the rest of the dependencies:
```bash
pip install -r requirements.txt
```




---

## ▶️ Run

```bash
python main.py --videoName video --videoFolder workdir
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

- Added `--minSpeechLen` to filter out short/non-speech segments.
- Skipped interpolation when face detections have no frame gaps, improving efficiency for continuous tracks.
- Applied weighted averaging across multi-duration inputs instead of repeating inference.
- Added `get_speaker_track_indices()` to isolate actual speaker tracks.

---


## Acknowledgements

This project builds on the great work from:

- [TalkNet-ASD](https://github.com/TaoRuijie/TalkNet-ASD) for active speaker detection.
- [YOLO-Face](https://github.com/akanametov/yolo-face) for face detection.


