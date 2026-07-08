import os
import types

import pytest

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample.mp4")
pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE), reason="tests/fixtures/sample.mp4 missing"
)


def _prep_video(tmp_path):
    """Runs extract_video/extract_audio into tmp_path, returns args stub."""
    from utils.video_utils import extract_audio, extract_video

    os.makedirs(tmp_path / "pyavi", exist_ok=True)
    os.makedirs(tmp_path / "pywork", exist_ok=True)
    args = types.SimpleNamespace(
        videoPath=FIXTURE,
        pyaviPath=str(tmp_path / "pyavi"),
        pyworkPath=str(tmp_path / "pywork"),
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
