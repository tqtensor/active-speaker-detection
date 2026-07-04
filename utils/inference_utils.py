import numpy

from config.logging_config import get_logger

logger = get_logger(__name__)


def get_speaker_track_indices(scores, args):
    """Identifies tracks with at least one speaking frame above threshold.

    Collects track indices that have any frame with speaking activity
    detected based on the configured threshold. Useful for identifying
    which tracks contain active speakers.

    Args:
        scores: List of per-frame speaking scores per track.
        args: Arguments containing the speakerThresh threshold value.

    Returns:
        List of track indices where speaker was detected.
    """
    speaker_track_indices = []
    for tidx, score in enumerate(scores):
        for fidx in range(len(score)):
            s = score[max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)]
            s = numpy.mean(s)
            if s >= args.speakerThresh:
                speaker_track_indices.append(tidx)
                break  # Only need to find one speaking frame
    return speaker_track_indices
