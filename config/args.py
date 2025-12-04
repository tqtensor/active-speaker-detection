import argparse

def get_args():
    parser = argparse.ArgumentParser(description="ASD demo")

    parser.add_argument('--videoName', type=str, default="video",
                        help='Demo video name')
    parser.add_argument('--videoFolder', type=str, default="workdir",
                        help='Path for inputs, tmps and outputs')
    parser.add_argument('--talkNetWeights', type=str, default="./weights/talknet/pretrain_TalkSet.model",
                        help='Path for the pretrained TalkNet model')
    parser.add_argument('--yoloFaceWeights', type=str, default="./weights/yolo/yolov11n-face.pt",
                        help='Path for the pretrained TalkNet model')
    parser.add_argument('--nDataLoaderThread', type=int, default=10,
                        help='Number of workers')
    parser.add_argument('--facedetScale', type=float, default=0.25,
                        help='Scale factor for face detection, the frames will be scale to 0.25 orig')
    parser.add_argument('--minTrack', type=int, default=10,
                        help='Number of min frames for each shot')
    parser.add_argument('--numFailedDet', type=int, default=10,
                        help='Number of missed detections allowed before tracking is stopped')
    parser.add_argument('--minFaceSize', type=int, default=1,
                        help='Minimum face size in pixels')
    parser.add_argument('--cropScale', type=float, default=0.40,
                        help='Scale bounding box')
    parser.add_argument('--start', type=int, default=0,
                        help='The start time of the video')
    parser.add_argument('--duration', type=int, default=0,
                        help='The duration of the video, when set as 0, will extract the whole video')
    parser.add_argument('--speakerThresh', type=float, default=0.6,
                        help='speaker detection threshold')
    parser.add_argument('--ignoreMultiSpeakers', type=bool, default=False,
                        help='ignores segments with multiple speakers')
    parser.add_argument('--minSpeechLen', type=float, default=0.25,
                        help='minimum speech length to be considered as a speaker')
    parser.add_argument('--yoloBatchSize', type=int, default=32,
                        help='Batch size for YOLO face detection (increase for more GPU utilization)')

    return parser.parse_args()
