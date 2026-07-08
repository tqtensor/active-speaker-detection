import numpy

from utils.compare import argmax_agreement, sign_agreement


def test_sign_agreement_counts_matching_signs():
    a = [[1.0, -1.0, 2.0]]
    b = [[3.0, -0.5, -1.0]]  # signs: (+,-,+) vs (+,-,-) -> 2/3 agree
    assert round(sign_agreement(a, b), 3) == 0.667


def _two_tracks():
    return [
        {"track": {"frame": numpy.array([0, 1])}},
        {"track": {"frame": numpy.array([0, 1])}},
    ]


def test_argmax_agreement_multi_face():
    tracks = _two_tracks()
    a = [[2.0, 2.0], [1.0, 1.0]]  # both frames: track 0 is top
    b = [[0.5, 0.0], [0.1, 1.0]]  # frame0: track0 top (agree); frame1: track1 top (disagree)
    assert argmax_agreement(tracks, a, b) == 0.5


def test_argmax_agreement_none_without_multiface():
    tracks = [{"track": {"frame": numpy.array([0, 1])}}]  # single track -> no ranking
    assert argmax_agreement(tracks, [[1.0, 2.0]], [[0.1, 0.2]]) is None
