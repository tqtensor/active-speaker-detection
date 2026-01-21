"""Fast video loading utilities using decord.

This module provides optimized video loading using decord library which is
2-3x faster than cv2.VideoCapture for batch frame reading. Falls back to
cv2 if decord is not available.
"""

import os
from typing import Optional

import cv2
import numpy as np

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


def load_video_features_decord(video_path: str) -> np.ndarray:
    """Load video and extract face features using decord (2-3x faster).

    Reads all frames at once using batch loading, then processes them
    to the TalkNet input format (grayscale, 112x112 center crop).

    Args:
        video_path: Path to the video file (.avi).

    Returns:
        numpy.ndarray: Array of shape (N, 112, 112) containing processed
        grayscale face frames.
    """
    if not DECORD_AVAILABLE:
        return load_video_features_cv2(video_path)

    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # (N, H, W, 3) RGB

    video_feature = []
    for frame in frames:
        # decord uses RGB, convert to grayscale
        face = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        face = cv2.resize(face, (224, 224))
        # Center crop 112x112
        face = face[56:168, 56:168]
        video_feature.append(face)

    return np.array(video_feature)


def load_video_features_cv2(video_path: str) -> np.ndarray:
    """Load video and extract face features using cv2 (fallback).

    This is the original implementation, kept for compatibility.

    Args:
        video_path: Path to the video file (.avi).

    Returns:
        numpy.ndarray: Array of shape (N, 112, 112) containing processed
        grayscale face frames.
    """
    video = cv2.VideoCapture(video_path)
    video_feature = []

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            face = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (224, 224))
            # Center crop 112x112
            face = face[
                int(112 - (112 / 2)): int(112 + (112 / 2)),
                int(112 - (112 / 2)): int(112 + (112 / 2)),
            ]
            video_feature.append(face)
        else:
            break

    video.release()
    return np.array(video_feature)


def preload_video_data(
    files: list,
    crop_path: str,
    use_decord: bool = True
) -> dict:
    """Preload all video data into memory for faster inference.

    Loads all video files upfront so the inference loop doesn't have
    I/O blocking in the hot path.

    Args:
        files: List of video file paths (.avi files).
        crop_path: Base path where cropped files are stored.
        use_decord: Whether to use decord (faster) or cv2 (fallback).

    Returns:
        dict: Mapping of filename (without extension) to video features array.
    """
    import tqdm
    from scipy.io import wavfile
    import python_speech_features

    preloaded = {}
    load_func = load_video_features_decord if (use_decord and DECORD_AVAILABLE) else load_video_features_cv2

    for file in tqdm.tqdm(files, desc="Pre-loading video data"):
        file_name = os.path.splitext(os.path.basename(file))[0]

        # Load video features
        video_features = load_func(file)

        # Load audio features
        audio_path = os.path.join(crop_path, file_name + ".wav")
        _, audio = wavfile.read(audio_path)
        audio_features = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010
        )

        preloaded[file_name] = {
            'video': video_features,
            'audio': audio_features
        }

    return preloaded


# Check if decord with GPU support is available
def check_decord_gpu() -> bool:
    """Check if decord GPU decoding is available.

    Returns:
        bool: True if GPU decoding is supported, False otherwise.
    """
    if not DECORD_AVAILABLE:
        return False

    try:
        from decord import gpu
        return True
    except ImportError:
        return False
