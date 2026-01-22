import os

import av
import cv2
import numpy as np


def load_video_features_pyav(video_path: str) -> np.ndarray:
    """Loads video and extracts face features using PyAV.

    Reads all frames using PyAV (faster than cv2), then processes them
    to the TalkNet input format (grayscale, 112x112 center crop).

    Args:
        video_path: Path to the video file (.avi).

    Returns:
        Array of shape (N, 112, 112) containing processed grayscale face
        frames.
    """
    container = av.open(video_path)
    video_feature = []

    for frame in container.decode(video=0):
        # PyAV uses RGB, convert to grayscale
        img = frame.to_ndarray(format="rgb24")
        face = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face = cv2.resize(face, (224, 224))
        # Center crop 112x112
        face = face[56:168, 56:168]
        video_feature.append(face)

    container.close()
    return np.array(video_feature)


def load_video_features_cv2(video_path: str) -> np.ndarray:
    """Loads video and extracts face features using cv2.

    Fallback implementation using OpenCV, kept for compatibility.

    Args:
        video_path: Path to the video file (.avi).

    Returns:
        Array of shape (N, 112, 112) containing processed grayscale face
        frames.
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
                int(112 - (112 / 2)) : int(112 + (112 / 2)),
                int(112 - (112 / 2)) : int(112 + (112 / 2)),
            ]
            video_feature.append(face)
        else:
            break

    video.release()
    return np.array(video_feature)


def preload_video_data(files: list, crop_path: str, use_pyav: bool = True) -> dict:
    """Preloads all video data into memory for faster inference.

    Loads all video files upfront so the inference loop doesn't have
    I/O blocking in the hot path.

    Args:
        files: List of video file paths (.avi files).
        crop_path: Base path where cropped files are stored.
        use_pyav: Whether to use PyAV (faster) or cv2 (fallback).

    Returns:
        Mapping of filename (without extension) to dict containing 'video'
        and 'audio' feature arrays.
    """
    import python_speech_features
    import tqdm
    from scipy.io import wavfile

    preloaded = {}
    load_func = load_video_features_pyav if use_pyav else load_video_features_cv2

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

        preloaded[file_name] = {"video": video_features, "audio": audio_features}

    return preloaded
