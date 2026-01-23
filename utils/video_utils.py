import os

import ffmpeg
import numpy
import python_speech_features
from scipy.io import wavfile

from config.logging_config import get_logger

logger = get_logger(__name__)


def extract_MFCC(file, outPath):
    """Extracts MFCC features from an audio file and saves them as a numpy array.

    Reads a WAV audio file, computes Mel-Frequency Cepstral Coefficients (MFCC),
    and saves the resulting feature matrix to a .npy file. The MFCC matrix has
    shape (N_frames, 13) where 1 second of audio corresponds to 100 frames.

    Args:
        file: Path to the input WAV audio file.
        outPath: Directory path where the output .npy file will be saved.
    """
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio, sr)  # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split("/")[-1].replace(".wav", ".npy"))
    numpy.save(featuresPath, mfcc)


def extract_video(args):
    """Extracts video from the source file and saves it as an AVI file.

    Processes the input video using ffmpeg to extract either the full video or
    a segment based on the args.duration parameter. The output is saved with
    specified quality settings and frame rate of 25 fps.

    Args:
        args: Configuration object containing:
            - videoPath: Path to the input video file.
            - pyaviPath: Output directory for the extracted video.
            - duration: Duration in seconds (0 for full video extraction).
            - start: Start time in seconds for segment extraction.
            - nDataLoaderThread: Number of threads for ffmpeg processing.

    Raises:
        ValueError: If start time or duration is negative.
        RuntimeError: If ffmpeg fails to extract the video.
    """
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
            logger.error(f"FFmpeg failed to extract video: {e.stderr.decode()}")
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
            logger.error(f"FFmpeg failed to extract video segment: {e.stderr.decode()}")
            raise RuntimeError(f"Failed to extract video segment: {e.stderr.decode()}")

    logger.info(f"Extracted video saved to {args.videoFilePath}")


def extract_audio(args):
    """Extracts audio from the video file and saves it as a WAV file.

    Converts the video file to a mono-channel WAV audio file at 16kHz sample
    rate using ffmpeg. The audio is extracted from the previously processed
    video file.

    Args:
        args: Configuration object containing:
            - videoFilePath: Path to the input video file.
            - pyaviPath: Output directory for the extracted audio.
            - nDataLoaderThread: Number of threads for ffmpeg processing.

    Raises:
        RuntimeError: If ffmpeg fails to extract the audio.
    """
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
        logger.error(f"FFmpeg failed to extract audio: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to extract audio: {e.stderr.decode()}")
    logger.info(f"Extracted audio saved to {args.audioFilePath}")


def extract_frames(args):
    """Extracts individual frames from the video file and saves them as JPEG images.

    Converts the video file into a sequence of JPEG image files using ffmpeg.
    Each frame is saved with a sequential numeric filename (e.g., 000001.jpg,
    000002.jpg, etc.). Optionally downscales frames to specified height.

    Args:
        args: Configuration object containing:
            - videoFilePath: Path to the input video file.
            - pyframesPath: Output directory for the extracted frame images.
            - nDataLoaderThread: Number of threads for ffmpeg processing.
            - frameHeight: Optional height to downscale frames (maintains aspect ratio).

    Raises:
        RuntimeError: If ffmpeg fails to extract the frames.
    """
    # Extract the video frames
    try:
        input_stream = ffmpeg.input(args.videoFilePath)

        # Apply scaling if frameHeight is specified
        if args.frameHeight:
            # Scale to specified height, maintaining aspect ratio (-1 for width)
            input_stream = input_stream.filter("scale", -1, args.frameHeight)
            logger.info(f"Downscaling frames to height={args.frameHeight}px")

        (
            input_stream.output(
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
        logger.error(f"FFmpeg failed to extract frames: {e.stderr.decode()}")
        raise RuntimeError(f"Failed to extract frames: {e.stderr.decode()}")
    logger.info(f"Extracted frames saved to {args.pyframesPath}")
