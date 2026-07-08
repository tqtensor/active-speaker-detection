import numpy

from config.logging_config import get_logger

logger = get_logger(__name__)


def sign_agreement(scores_a, scores_b):
    """Fraction of matched (track, frame) score pairs that agree on sign.

    Scale-invariant: only the sign (speaking-ish vs not) is compared, so it is
    valid across models whose score magnitudes differ. Tracks are matched by
    index; per track, only the overlapping frame prefix is compared.
    """
    total = 0
    agree = 0
    for sa, sb in zip(scores_a, scores_b):
        sa = numpy.asarray(sa, dtype=float)
        sb = numpy.asarray(sb, dtype=float)
        n = min(len(sa), len(sb))
        if n == 0:
            continue
        total += n
        agree += int(numpy.sum(numpy.sign(sa[:n]) == numpy.sign(sb[:n])))
    return (agree / total) if total else 0.0


def _frame_to_track_scores(tracks, scores):
    """Build frame_id -> list of (track_index, score_at_frame)."""
    frame_map = {}
    for tidx, track in enumerate(tracks):
        frames = track["track"]["frame"].tolist()
        s = numpy.asarray(scores[tidx], dtype=float)
        for fidx, frame in enumerate(frames):
            if fidx < len(s):
                frame_map.setdefault(int(frame), []).append((tidx, float(s[fidx])))
    return frame_map


def argmax_agreement(tracks, scores_a, scores_b):
    """Fraction of multi-face frames where both models pick the same top track.

    Threshold-free: compares which track is most-likely-speaking per frame, not
    absolute score values, so it is valid across differently-scaled models.
    Frames with fewer than two faces under either model are ignored. Returns
    None if no comparable multi-face frame exists.
    """
    fa = _frame_to_track_scores(tracks, scores_a)
    fb = _frame_to_track_scores(tracks, scores_b)
    considered = 0
    agree = 0
    for frame, a_list in fa.items():
        b_list = fb.get(frame)
        if len(a_list) < 2 or not b_list or len(b_list) < 2:
            continue
        top_a = max(a_list, key=lambda p: p[1])[0]
        top_b = max(b_list, key=lambda p: p[1])[0]
        considered += 1
        agree += int(top_a == top_b)
    if considered == 0:
        return None
    return agree / considered
