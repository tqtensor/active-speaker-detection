# tests/test_tracks_deterministic.py
"""Tracks must be identical across ASD models: detection+tracking are upstream
of the model branch, so a TalkNet run and a LightASD run should agree on every
track. This underpins the per-(track, frame) joins in the speed and agreement
tests.
"""

import os
import pickle
import subprocess

import numpy
import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:
    CUDA = False

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
pytestmark = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def _run_tracks(model):
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "main.py",
            "--videoName",
            "sample",
            "--videoFolder",
            FIX,
            "--asdModel",
            model,
            "--metadataOnly",
        ],
        check=True,
    )
    with open(os.path.join(FIX, "sample", "pywork", "tracks.pckl"), "rb") as f:
        return pickle.load(f)


def test_tracks_identical_across_models(sample_video):
    talk = _run_tracks("talknet")
    light = _run_tracks("lightasd")
    assert len(talk) == len(light), "track count differs between models"
    for i, (a, b) in enumerate(zip(talk, light)):
        assert numpy.array_equal(a["track"]["frame"], b["track"]["frame"]), (
            f"track {i} frame indices differ"
        )
        assert numpy.allclose(a["track"]["bbox"], b["track"]["bbox"]), (
            f"track {i} bboxes differ"
        )
