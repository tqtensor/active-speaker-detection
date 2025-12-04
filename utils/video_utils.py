import os
import sys
import time

import ffmpeg
import numpy
import python_speech_features
from scipy.io import wavfile


def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio, sr)  # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split("/")[-1].replace(".wav", ".npy"))
    numpy.save(featuresPath, mfcc)


def extract_video(args):
    # Extract video
    args.videoFilePath = os.path.join(args.pyaviPath, "video.avi")
    # Extract full video or a segment depending on args.duration
    if args.duration == 0:
        try:
            stream = (
                ffmpeg.input(args.videoPath)
                .output(
                    args.videoFilePath,
                    # vcodec='mpeg4',  # Explicitly set video codec
                    # acodec='copy',  # Copy audio stream
                    qscale=2,  # Quality scale
                    r=25,  # Frame rate
                    # async_='1',
                    threads=args.nDataLoaderThread,
                )
                .global_args("-loglevel", "error")  # Set loglevel as global arg
                .overwrite_output()
            )
            # Run the ffmpeg command
            out, err = stream.run(capture_stdout=True, capture_stderr=True)

        except ffmpeg.Error as e:
            print(f"FFmpeg stderr:\n{e.stderr.decode()}")
            raise RuntimeError(f"Failed to extract full video: {e.stderr.decode()}")
    # else extract a segment of the video:
    else:
        if args.start < 0:
            raise ValueError("Start time must be non-negative")
        if args.duration < 0:
            raise ValueError("Duration must be non-negative")

        try:
            stream = (
                ffmpeg.input(args.videoPath)
                .output(
                    args.videoFilePath,
                    # vcodec='mpeg4',  # Explicitly set video codec
                    # acodec='copy',  # Copy audio stream
                    qscale=2,  # Quality scale
                    r=25,  # Frame rate
                    # async_='1',
                    ss=args.start,  # Start time
                    t=args.duration,  # Duration
                    threads=args.nDataLoaderThread,
                )
                .global_args("-loglevel", "error")  # Set loglevel as global arg
                .overwrite_output()
            )
            # Run the ffmpeg command
            out, err = stream.run(capture_stdout=True, capture_stderr=True)

        except ffmpeg.Error as e:
            print(f"FFmpeg stderr:\n{e.stderr.decode()}")
            raise RuntimeError(f"Failed to extract video segment: {e.stderr.decode()}")

    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Extract the video and save in %s \r\n" % (args.videoFilePath)
    )


def extract_audio(args):
    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, "audio.wav")
    try:
        (
            ffmpeg.input(args.videoFilePath)
            .output(
                args.audioFilePath,
                ac=1,  # Mono channel
                ar=16000,  # Sample rate
                qscale="0",  # Audio quality
                vn=None,  # No video
                threads=args.nDataLoaderThread,
            )
            .global_args("-loglevel", "error")  # Silence ffmpeg logs
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg stderr:\n{e.stderr.decode()}")
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Extract the audio and save in %s \r\n" % (args.audioFilePath)
    )


def extract_frames(args):
    # Extract the video frames
    try:
        (
            ffmpeg.input(args.videoFilePath)
            .output(
                os.path.join(args.pyframesPath, "%06d.jpg"),
                qscale="2",  # Image quality
                f="image2",  # Image muxer
                threads=args.nDataLoaderThread,
            )
            .global_args("-loglevel", "error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"FFmpeg stderr:\n{e.stderr.decode()}")
        raise RuntimeError(f"Failed to extract frames: {e.stderr.decode()}")
    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + " Extract the frames and save in %s \r\n" % (args.pyframesPath)
    )
