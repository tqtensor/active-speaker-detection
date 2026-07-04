import torch

from utils.gpu_crop import extract_talknet_features_gpu


def test_extract_talknet_features_gpu_output_shape():
    """Output must be (T, 112, 112) - grayscale, center-cropped from 224."""
    crops = torch.full((2, 3, 224, 224), 0.5)
    gray = extract_talknet_features_gpu(crops)
    assert gray.shape == (2, 112, 112)


def test_extract_talknet_features_gpu_scales_to_0_255():
    """Model frontends (Light-ASD, TalkNet) divide by 255 internally, so this
    helper must output raw pixel values in [0, 255], not [0, 1].

    A constant 0.5-valued RGB input represents a mid-gray pixel in [0, 1]
    space; the correct [0, 255]-scaled output is ~127.5, not ~0.5.
    """
    crops = torch.full((2, 3, 224, 224), 0.5)
    gray = extract_talknet_features_gpu(crops)

    assert gray.max().item() > 100.0, (
        "output looks like it is still in [0, 1] range; expected [0, 255]"
    )
    assert gray.min().item() >= 0.0


def test_extract_talknet_features_gpu_luminance_correctness():
    """For an all-0.5 RGB input, luminance = (0.299+0.587+0.114)*0.5 = 0.5,
    scaled to [0, 255] gives ~127.5.
    """
    crops = torch.full((2, 3, 224, 224), 0.5)
    gray = extract_talknet_features_gpu(crops)

    expected = 0.5 * 255.0
    assert torch.allclose(gray, torch.full_like(gray, expected), atol=1e-3)


def test_extract_talknet_features_gpu_ramp_luminance():
    """Sanity check luminance weighting + scaling on a non-constant input."""
    crops = torch.zeros((1, 3, 224, 224))
    crops[:, 0] = 0.2  # R
    crops[:, 1] = 0.4  # G
    crops[:, 2] = 0.6  # B

    gray = extract_talknet_features_gpu(crops)

    expected = (0.299 * 0.2 + 0.587 * 0.4 + 0.114 * 0.6) * 255.0
    assert torch.allclose(gray, torch.full_like(gray, expected), atol=1e-3)
    # Still in [0, 255], well above the [0, 1] range.
    assert gray.max().item() > 50.0
