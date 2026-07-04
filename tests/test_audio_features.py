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
