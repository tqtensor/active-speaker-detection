import os
import pickle

import numpy
from scenedetect import ContentDetector, SceneManager, open_video
from scipy.interpolate import interp1d

from config.logging_config import get_logger

logger = get_logger(__name__)


def scene_detect(args):
    """Detects scene cuts in the video and saves them to a pickle file.

    Analyzes the video for scene changes using content-based detection.
    If no scene cuts are detected, treats the entire video as one scene.

    Args:
        args: Configuration object with videoFilePath and pyworkPath attributes.

    Returns:
        List of tuples containing (start_timecode, end_timecode) for each scene.
    """
    # CPU: Scene detection, output is the list of each shot's time duration
    video = open_video(args.videoFilePath)
    sceneManager = SceneManager()
    sceneManager.add_detector(ContentDetector())
    sceneManager.detect_scenes(video)
    sceneList = sceneManager.get_scene_list()
    savePath = os.path.join(args.pyworkPath, "scene.pckl")
    if sceneList == []:
        # If no scene cuts detected, treat entire video as one scene
        sceneList = [(video.base_timecode, video.current_timecode)]
    with open(savePath, "wb") as fil:
        pickle.dump(sceneList, fil)
        logger.info(f"{args.videoFilePath} - scenes detected {len(sceneList)}")
    return sceneList


def bb_intersection_over_union(boxA, boxB, evalCol=False):
    """Calculates intersection over union (IOU) between two bounding boxes.

    Computes the overlap ratio between two rectangular bounding boxes.
    Used for tracking faces across consecutive frames.

    Args:
        boxA: First bounding box as [x1, y1, x2, y2].
        boxB: Second bounding box as [x1, y1, x2, y2].
        evalCol: If True, calculates intersection/boxA. If False, calculates
            standard IOU (intersection/union).

    Returns:
        Overlap ratio as a float between 0 and 1.
    """
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def track_shot(args, sceneFaces):
    """Tracks faces across frames within a scene using IOU-based matching.

    Groups face detections across consecutive frames into continuous tracks
    by matching bounding boxes. Interpolates missing detections and filters
    tracks based on length and face size criteria.

    Args:
        args: Configuration object with numFailedDet, minTrack, and minFaceSize
            attributes.
        sceneFaces: List of lists, where each inner list contains face detections
            for a single frame. Each face is a dict with 'frame' and 'bbox' keys.
            This list is modified in-place (faces are removed as they're assigned).

    Returns:
        List of track dictionaries, each containing 'frame' (array of frame indices)
        and 'bbox' (array of interpolated bounding boxes).
    """
    # CPU: Face tracking
    iouThres = 0.5  # Minimum IOU between consecutive face detections
    tracks = []
    while True:
        track = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                # find distance between the last registered face (track[-1]) and the current face.
                elif face["frame"] - track[-1]["frame"] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum = numpy.array([f["frame"] for f in track])
            bboxes = numpy.array([numpy.array(f["bbox"]) for f in track])
            frameI = numpy.arange(frameNum[0], frameNum[-1] + 1)

            ## check if frameI length is equal to frameNum length (e.g. no gaps), then no need to interpolate. More efficient.
            if len(frameI) == len(frameNum):
                bboxesI = bboxes  # no gaps, no need to interpolate
            else:
                bboxesI = []
                for ij in range(0, 4):
                    # interpolate the bbox for each frame to fill the gaps
                    interpfn = interp1d(frameNum, bboxes[:, ij])
                    bboxesI.append(interpfn(frameI))
                bboxesI = numpy.stack(bboxesI, axis=1)

            if (
                max(
                    numpy.mean(bboxesI[:, 2] - bboxesI[:, 0]),
                    numpy.mean(bboxesI[:, 3] - bboxesI[:, 1]),
                )
                > args.minFaceSize
            ):
                tracks.append({"frame": frameI, "bbox": bboxesI})
    return tracks
