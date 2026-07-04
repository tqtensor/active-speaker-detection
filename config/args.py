import argparse


def get_args():
    """Parses command-line arguments for active speaker detection.

    Configures and returns arguments for video processing, model weights,
    face detection, tracking parameters, and visualization options.

    Returns:
        Namespace object containing all parsed command-line arguments including
        video paths, model configurations, detection thresholds, and processing
        options.
    """
    parser = argparse.ArgumentParser(description="ASD demo")

    parser.add_argument(
        "--videoName", type=str, default="video", help="Demo video name"
    )
    parser.add_argument(
        "--videoFolder",
        type=str,
        default="workdir",
        help="Path for inputs, tmps and outputs",
    )
    parser.add_argument(
        "--talkNetWeights",
        type=str,
        default="./weights/talknet/pretrain_TalkSet.model",
        help="Path for the pretrained TalkNet model",
    )
    parser.add_argument(
        "--asdModel",
        type=str,
        default="lightasd",
        choices=["talknet", "lightasd"],
        help="Active speaker model: lightasd (default, fast) or talknet",
    )
    parser.add_argument(
        "--lightAsdWeights",
        type=str,
        default="./weights/lightasd/pretrain_AVA_CVPR.model",
        help="Path for the pretrained Light-ASD model",
    )
    parser.add_argument(
        "--yoloVariant",
        type=str,
        default="n",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv11 variant for face detection: n (nano, fastest), s (small), "
        "m (medium), l (large), x (extra-large, best accuracy)",
    )
    parser.add_argument(
        "--yoloFaceWeights",
        type=str,
        default=None,
        help="Path for the YOLO face detection model (auto-set based on --yoloVariant if not provided)",
    )
    parser.add_argument(
        "--nDataLoaderThread", type=int, default=10, help="Number of workers"
    )
    parser.add_argument(
        "--facedetScale",
        type=float,
        default=0.25,
        help="Scale factor for face detection, the frames will be scale to 0.25 orig",
    )
    parser.add_argument(
        "--minTrack", type=int, default=10, help="Number of min frames for each shot"
    )
    parser.add_argument(
        "--numFailedDet",
        type=int,
        default=10,
        help="Number of missed detections allowed before tracking is stopped",
    )
    parser.add_argument(
        "--minFaceSize", type=int, default=1, help="Minimum face size in pixels"
    )
    parser.add_argument(
        "--cropScale", type=float, default=0.40, help="Scale bounding box"
    )
    parser.add_argument(
        "--start", type=int, default=0, help="The start time of the video"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="The duration of the video, when set as 0, will extract the whole video",
    )
    parser.add_argument(
        "--speakerThresh", type=float, default=0.6, help="speaker detection threshold"
    )
    parser.add_argument(
        "--ignoreMultiSpeakers",
        type=bool,
        default=False,
        help="ignores segments with multiple speakers",
    )
    parser.add_argument(
        "--minSpeechLen",
        type=float,
        default=0.25,
        help="minimum speech length to be considered as a speaker",
    )
    parser.add_argument(
        "--yoloBatchSize",
        type=int,
        default=32,
        help="Batch size for YOLO face detection (increase for more GPU utilization)",
    )
    parser.add_argument(
        "--jpegQscale",
        type=int,
        default=2,
        help="JPEG quality scale for extracted frames (1-31, lower=better quality). Default: 2",
    )
    parser.add_argument(
        "--metadataOnly",
        action="store_true",
        help="Skip visualization video, only produce frame_metadata.json and speaker_summary.json",
    )
    parser.add_argument(
        "--useBatched",
        action="store_true",
        help="Use batched TalkNet inference for faster processing (2-3x speedup)",
    )
    parser.add_argument(
        "--talknetBatchSize",
        type=int,
        default=16,
        help="Batch size for TalkNet inference when using --useBatched",
    )
    parser.add_argument(
        "--logLevel",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level (default: INFO)",
    )
    parser.add_argument(
        "--logFile",
        type=str,
        default=None,
        help="Path to log file (default: None, console only)",
    )
    parser.add_argument(
        "--noColor",
        action="store_true",
        help="Disable colored console output",
    )

    return parser.parse_args()
