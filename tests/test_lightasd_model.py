import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_lightasd_forward_shapes():
    from model.lightASD import lightASD

    net = lightASD()
    net.eval()
    T = 25
    v = torch.rand(1, T, 112, 112).cuda() * 255  # raw 0-255 grayscale
    a = torch.rand(1, T * 4, 13).cuda()

    with torch.no_grad():
        ea = net.model.forward_audio_frontend(a)
        ev = net.model.forward_visual_frontend(v)
        out = net.model.forward_audio_visual_backend(ea, ev)
        score = net.lossAV.forward(out, labels=None)

    score = np.asarray(score).reshape(-1)
    assert score.shape[0] == T
    assert np.isfinite(score).all()
