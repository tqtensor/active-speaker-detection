import math
import os
import sys

import numpy
import torch
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.talkNet import talkNet


def evaluate_network(files, args):
    """Evaluates active speaker detection using pretrained TalkNet.

    Processes audio-visual files through TalkNet to compute speaking scores.
    Uses multiple duration windows with weighted averaging for robust inference.
    This optimized version uses decord for faster video loading and pre-loads
    all data before the inference loop to minimize I/O in the hot path.

    Args:
        files: List of file paths to evaluate.
        args: Arguments containing pycropPath and talkNetWeights attributes.

    Returns:
        List of per-frame speaking scores for each input file.
    """
    from utils.video_loader import preload_video_data

    # GPU: active speaker detection by pretrained TalkNet
    s = talkNet()
    s.loadParameters(args.talkNetWeights)
    sys.stderr.write("Model %s loaded from previous state! \r\n" % args.talkNetWeights)
    s.eval()

    # Pre-load all video/audio data for faster inference loop
    preloaded = preload_video_data(files, args.pycropPath, use_decord=True)

    allScores = []

    # Define duration set and weights (higher weight = more influence)
    durationSet = [1, 2, 3, 4, 5, 6]
    weights = [3, 3, 2, 1, 1, 1]

    for file in tqdm.tqdm(files, total=len(files), desc="Running TalkNet inference"):
        fileName = os.path.splitext(os.path.basename(file))[0]

        # Get pre-loaded data (no I/O in hot path)
        data = preloaded[fileName]
        audioFeature = data["audio"]
        videoFeature = data["video"]

        # Align audio and video lengths
        length = min(
            (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
            videoFeature.shape[0] / 25,
        )
        audioFeature = audioFeature[: int(round(length * 100)), :]
        videoFeature = videoFeature[: int(round(length * 25)), :, :]

        allScore = []
        for idx, duration in enumerate(durationSet):
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = (
                        torch.FloatTensor(
                            audioFeature[
                                i * duration * 100 : (i + 1) * duration * 100, :
                            ]
                        )
                        .unsqueeze(0)
                        .cuda()
                    )
                    inputV = (
                        torch.FloatTensor(
                            videoFeature[
                                i * duration * 25 : (i + 1) * duration * 25, :, :
                            ]
                        )
                        .unsqueeze(0)
                        .cuda()
                    )
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)

            # Apply weight by repeating scores
            for _ in range(weights[idx]):
                allScore.append(scores)

        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(
            float
        )
        allScores.append(allScore)

    return allScores


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


def evaluate_network_batched(files, args, batch_size=16):
    """Evaluates active speaker detection using batched TalkNet inference.

    Processes multiple face tracks in parallel using DataLoader for
    2-3x speedup over sequential processing. Uses decord for fast
    video loading and pre-loads all data before inference to minimize
    I/O overhead in the hot path.

    Args:
        files: List of file paths to evaluate.
        args: Arguments containing pycropPath and talkNetWeights attributes.
        batch_size: Number of track segments to process per batch.

    Returns:
        List of per-frame speaking scores for each input file.
    """
    from collections import defaultdict

    from utils.dataset import create_dataloader
    from utils.video_loader import preload_video_data

    # Load TalkNet model
    s = talkNet()
    s.loadParameters(args.talkNetWeights)
    sys.stderr.write("Model %s loaded from previous state! \r\n" % args.talkNetWeights)
    s.eval()

    # Pre-load all video/audio data
    preloaded = preload_video_data(files, args.pycropPath, use_decord=True)

    # Extract features as dicts
    video_features = {}
    audio_features = {}
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        data = preloaded[file_name]

        # Align lengths
        af = data["audio"]
        vf = data["video"]
        length = min(
            (af.shape[0] - af.shape[0] % 4) / 100,
            vf.shape[0] / 25,
        )
        video_features[file_name] = vf[: int(round(length * 25)), :, :]
        audio_features[file_name] = af[: int(round(length * 100)), :]

    # Process each duration with batching
    durationSet = [1, 2, 3, 4, 5, 6]
    weights = [3, 3, 2, 1, 1, 1]

    all_scores = {fn: [] for fn in video_features.keys()}

    for dur_idx, duration in enumerate(durationSet):
        loader = create_dataloader(
            video_features,
            audio_features,
            duration=duration,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
        )

        # Store scores by (track_id, segment_idx)
        segment_scores = defaultdict(dict)

        with torch.no_grad():
            for batch in tqdm.tqdm(loader, desc=f"Duration {duration}s"):
                videos = batch["videos"].cuda()
                audios = batch["audios"].cuda()

                # Forward pass through TalkNet
                embedA = s.model.forward_audio_frontend(audios)
                embedV = s.model.forward_visual_frontend(videos)
                embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                out = s.model.forward_audio_visual_backend(embedA, embedV)
                batch_scores = s.lossAV.forward(out, labels=None)

                # Store scores by track and segment
                for i, (tid, seg_idx, vid_len) in enumerate(
                    zip(batch["track_ids"], batch["seg_idxs"], batch["video_lengths"])
                ):
                    # Extract only valid scores (up to actual length)
                    score = batch_scores[i]
                    if isinstance(score, list):
                        segment_scores[tid][seg_idx] = score
                    else:
                        segment_scores[tid][seg_idx] = [score]

        # Aggregate scores per track and apply weights
        for tid in video_features.keys():
            if tid not in segment_scores:
                continue
            # Combine segments in order
            track_scores = []
            for seg_idx in sorted(segment_scores[tid].keys()):
                track_scores.extend(segment_scores[tid][seg_idx])

            # Apply weight by repeating
            for _ in range(weights[dur_idx]):
                all_scores[tid].append(track_scores)

    # Final averaging - maintain same order as input files
    final_scores = []
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        if file_name in all_scores and all_scores[file_name]:
            scores = numpy.round(
                numpy.mean(numpy.array(all_scores[file_name]), axis=0), 1
            ).astype(float)
            final_scores.append(scores)
        else:
            final_scores.append(numpy.array([]))

    return final_scores
