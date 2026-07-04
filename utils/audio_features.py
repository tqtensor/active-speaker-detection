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
