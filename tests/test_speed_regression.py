# tests/test_speed_regression.py
"""Relative speed-regression gate. Latency is hardware-dependent, so we assert a
PORTABLE quantity: the ratio of TalkNet's asd-stage time to LightASD's, against
a baseline captured on this box.

Capture / refresh the baseline on a GPU box:
    uv run python main.py --videoName sample --videoFolder tests/fixtures \
        --asdModel talknet  --metadataOnly
    T=$(python -c "import json;print(json.load(open('tests/fixtures/sample/pywork/timings.json'))['stages']['asd']['seconds'])")
    uv run python main.py --videoName sample --videoFolder tests/fixtures \
        --asdModel lightasd --metadataOnly
    L=$(python -c "import json;print(json.load(open('tests/fixtures/sample/pywork/timings.json'))['stages']['asd']['seconds'])")
    python -c "import json;json.dump({'asd_ratio_talknet_over_lightasd': $T/$L}, open('tests/fixtures/speed_baseline.json','w'))"
"""

import json
import os
import subprocess

import pytest

try:
    import torch

    CUDA = torch.cuda.is_available()
except ImportError:
    CUDA = False

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
BASELINE = os.path.join(FIX, "speed_baseline.json")
TOLERANCE = 0.25  # ±25%: allow noise, catch a real regression

pytestmark = pytest.mark.skipif(not CUDA, reason="needs CUDA")


def _asd_seconds(model):
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
    with open(os.path.join(FIX, "sample", "pywork", "timings.json")) as f:
        return json.load(f)["stages"]["asd"]["seconds"]


def test_asd_speed_ratio_within_tolerance(sample_video):
    if not os.path.exists(BASELINE):
        pytest.skip("no speed_baseline.json; capture it on this box (see docstring)")
    ratio = _asd_seconds("talknet") / _asd_seconds("lightasd")
    base = json.load(open(BASELINE))["asd_ratio_talknet_over_lightasd"]
    drift = abs(ratio - base) / base
    assert drift <= TOLERANCE, (
        f"asd talknet/lightasd ratio {ratio:.3f} drifted {drift:.0%} from "
        f"baseline {base:.3f} (> {TOLERANCE:.0%})"
    )
