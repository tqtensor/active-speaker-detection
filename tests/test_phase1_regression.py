"""
Phase-1 regression gate: compare speaking decisions vs TalkNet baseline.

DEFERRED: Baseline capture (Step 1)
==================================
The baseline_talknet_metadata.json does not exist on CPU machines.
To capture a baseline on a GPU machine:

1. Ensure the fixture sample.mp4 exists at tests/fixtures/sample.mp4
2. Run the master branch pipeline on the fixture:
   cd /path/to/repo
   git checkout master  # or the stable branch before Phase 1 changes
   uv run python main.py \
       --videoName sample \
       --videoFolder tests/fixtures \
       --asdModel talknet \
       --metadataOnly
3. Copy the generated metadata to the baseline:
   cp tests/fixtures/sample/pywork/frame_metadata.json \
      tests/fixtures/baseline_talknet_metadata.json
4. Return to the feature branch and run the test:
   git checkout -
   uv run pytest tests/test_phase1_regression.py -v

Expected result: PASS (speaking decision agreement >= 99%)
"""

import json
import os
import subprocess

import pytest

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
BASELINE = os.path.join(FIX, "baseline_talknet_metadata.json")
pytestmark = pytest.mark.skipif(
    not (os.path.exists(os.path.join(FIX, "sample.mp4")) and os.path.exists(BASELINE)),
    reason="fixture clip or baseline metadata missing",
)


def _speaking_decisions(meta):
    out = {}
    for fr in meta["frames"]:
        for face in fr["faces"]:
            out[(fr["frame_id"], face["track_id"])] = face["is_speaking"]
    return out


def test_phase1_matches_baseline(tmp_path):
    subprocess.run(
        [
            "uv", "run", "python", "main.py",
            "--videoName", "sample", "--videoFolder", FIX,
            "--asdModel", "talknet", "--metadataOnly",
        ],
        check=True,
    )
    with open(os.path.join(FIX, "sample", "pywork", "frame_metadata.json")) as f:
        new = json.load(f)
    with open(BASELINE) as f:
        base = json.load(f)

    nd, bd = _speaking_decisions(new), _speaking_decisions(base)
    # Same set of (frame, track) speaking decisions >= 99% agreement.
    common = set(nd) & set(bd)
    assert common, "no overlapping (frame, track) keys"
    agree = sum(1 for k in common if nd[k] == bd[k]) / len(common)
    assert agree >= 0.99, f"speaking-decision agreement {agree:.3f} < 0.99"
