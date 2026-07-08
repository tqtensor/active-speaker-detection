# tests/test_cross_model_agreement.py
"""Cross-model agreement guardrail (NOT a quality metric).

The two ASD models emit different score SCALES and offsets, so only rank-based,
threshold-free agreement is meaningful. Per-frame speaker argmax (which face is
the top speaker) is the sound scale-invariant signal; it is asserted as a loose
tripwire anchored BELOW the observed baseline (~0.70 on the shared clip), so it
fires on a real drop in speaker-choice agreement, not on the two models'
inherent diversity. Sign agreement on raw scores is confounded by inter-model
score offset (measured 0.341 — below chance — because it captures the offset
gap, not who is speaking), so it is recorded for information only, never
asserted.
"""

import os
import pickle
import subprocess

import pytest

from utils.compare import argmax_agreement, sign_agreement

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:
    CUDA = False

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
# Observed argmax agreement on the shared clip is ~0.70; anchor the floor below
# it so the guardrail catches a regression in speaker-choice agreement rather
# than the models' inherent ~30% diversity.
ARGMAX_MIN = 0.6

pytestmark = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def _run(model):
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
    pw = os.path.join(FIX, "sample", "pywork")
    with open(os.path.join(pw, "scores.pckl"), "rb") as f:
        scores = pickle.load(f)
    with open(os.path.join(pw, "tracks.pckl"), "rb") as f:
        tracks = pickle.load(f)
    return scores, tracks


def test_models_agree_on_speaker_ranking(sample_video):
    scores_talk, tracks = _run("talknet")
    scores_light, _ = _run("lightasd")

    # Informational only: sign-of-raw-score is confounded by inter-model offset.
    sa = sign_agreement(scores_talk, scores_light)
    print(f"\n[cross-model] sign_agreement={sa:.3f} (informational, not asserted)")

    am = argmax_agreement(tracks, scores_talk, scores_light)
    print(f"[cross-model] argmax_agreement={am}")
    if am is not None:  # only assert when multi-face frames exist
        assert am >= ARGMAX_MIN, f"argmax agreement {am:.3f} < {ARGMAX_MIN}"
