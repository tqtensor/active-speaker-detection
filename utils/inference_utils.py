import math
import os
import sys

import cv2
import numpy
import python_speech_features
import torch
import tqdm
from scipy.io import wavfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.talkNet import talkNet


def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet
    s = talkNet()
    s.loadParameters(args.talkNetWeights)
    sys.stderr.write("Model %s loaded from previous state! \r\n" % args.talkNetWeights)
    s.eval()
    allScores = []

    # define the duration set and weights for each duration (higher weight = more influence). Used to infer from different durations (results are averaged).
    durationSet = [1, 2, 3, 4, 5, 6]
    weights = [3, 3, 2, 1, 1, 1]  # Higher weight = more influence

    # print('durationSet', durationSet)
    # raise Exception('Stop the code here')
    for file in tqdm.tqdm(files, total=len(files)):
        # print('file', file)

        fileName = os.path.splitext(os.path.basename(file))[0]  # Load audio and video
        # print('fileName', fileName)
        # print('os.path.join(args.pycropPath, fileName + .wav)', os.path.join(args.pycropPath, fileName + '.wav'))
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + ".wav"))
        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010
        )

        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + ".avi"))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                ]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min(
            (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
            videoFeature.shape[0] / 25,
        )
        audioFeature = audioFeature[: int(round(length * 100)), :]
        videoFeature = videoFeature[: int(round(length * 25)), :, :]
        allScore = []  # Evaluation use TalkNet
        # for duration in durationSet:
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
                    # print('scores', scores)

            # Apply weight by repeating scores (more efficient than repeated inference)
            for _ in range(weights[idx]):
                allScore.append(scores)
        allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(
            float
        )
        # print('allScore', allScore)
        allScores.append(allScore)
        # raise Exception('Stop the code here')
    return allScores


# improved by adding this function to get_speaker_track_indices (helps to focus on speaker tracks/IDs only)
def get_speaker_track_indices(scores, args):
    """
    Identify tracks with at least one speaking frame above threshold. i.e., Collect tracks that have
     any frame speaking (helps identify speakers).

    Args:
        scores: List of per-frame speaking scores per track.
        args: Arguments containing the speakerThresh value.

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
