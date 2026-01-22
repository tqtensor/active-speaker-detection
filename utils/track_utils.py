import glob
import os
import pickle
import sys

import cv2
import ffmpeg
import numpy
from scenedetect import ContentDetector, SceneManager, open_video
from scipy import signal
from scipy.interpolate import interp1d


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
        sys.stderr.write(
            "%s - scenes detected %d\n" % (args.videoFilePath, len(sceneList))
        )
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


def extract_audio_only(args, track, cropFile):
    """Extracts audio segment for a face track without video cropping.

    Optimized for GPU mode where video cropping is handled separately.
    Computes smoothed face detection coordinates and extracts the corresponding
    audio segment from the source audio file.

    Args:
        args: Configuration object with audioFilePath attribute.
        track: Track dictionary containing 'frame' and 'bbox' arrays.
        cropFile: Base path for output files (used for .wav output).

    Returns:
        Dictionary with 'track' (original track) and 'proc_track' (smoothed
        detection coordinates with 'x', 'y', 's' keys).
    """
    # Compute smoothed detection coordinates (needed for proc_track)
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)
        dets["x"].append((det[0] + det[2]) / 2)
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)

    # Extract audio segment
    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25

    (
        ffmpeg.input(args.audioFilePath, ss=audioStart, to=audioEnd)
        .output(
            audioTmp,
            ac=1,
            vn=None,
            acodec="pcm_s16le",
            ar=16000,
            threads=1,  # Use single thread per process (parallelism comes from multiprocessing)
            loglevel="panic",
        )
        .overwrite_output()
        .run()
    )

    return {"track": track, "proc_track": dets}


def crop_video(args, track, cropFile):
    """Crops face regions from video frames and extracts corresponding audio.

    Reads video frames, applies smoothing to face bounding boxes, crops and
    resizes face regions to 224x224, and writes to an AVI file. Also extracts
    the corresponding audio segment.

    Args:
        args: Configuration object with pyframesPath, cropScale, audioFilePath,
            and nDataLoaderThread attributes.
        track: Track dictionary containing 'frame' and 'bbox' arrays.
        cropFile: Base path for output files (used for .avi and .wav outputs).

    Returns:
        Dictionary with 'track' (original track) and 'proc_track' (smoothed
        detection coordinates with 'x', 'y', 's' keys).
    """
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(args.pyframesPath, "*.jpg"))  # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(
        cropFile + "t.avi", cv2.VideoWriter_fourcc(*"XVID"), 25, (224, 224)
    )  # Write video
    dets = {"x": [], "y": [], "s": []}
    for det in track["bbox"]:  # Read the tracks
        dets["s"].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets["y"].append((det[1] + det[3]) / 2)  # crop center x
        dets["x"].append((det[0] + det[2]) / 2)  # crop center y
    dets["s"] = signal.medfilt(dets["s"], kernel_size=13)  # Smooth detections
    dets["x"] = signal.medfilt(dets["x"], kernel_size=13)
    dets["y"] = signal.medfilt(dets["y"], kernel_size=13)
    for fidx, frame in enumerate(track["frame"]):
        cs = args.cropScale
        bs = dets["s"][fidx]  # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
        image = cv2.imread(flist[frame])
        frame = numpy.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(110, 110),
        )
        my = dets["y"][fidx] + bsi  # BBox center Y
        mx = dets["x"][fidx] + bsi  # BBox center X
        face = frame[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp = cropFile + ".wav"
    audioStart = (track["frame"][0]) / 25
    audioEnd = (track["frame"][-1] + 1) / 25
    vOut.release()

    # Extract audio segment for TalkNet inference
    (
        ffmpeg.input(
            args.audioFilePath, ss=audioStart, to=audioEnd
        )  # Load input audio and trim from start to end
        .output(
            audioTmp,  # Output file path
            ac=1,  # Set audio channels to mono
            vn=None,  # Disable video stream
            acodec="pcm_s16le",  # Set audio codec to 16-bit PCM
            ar=16000,  # Resample audio to 16 kHz
            threads=args.nDataLoaderThread,  # Use specified number of threads
            loglevel="panic",  # Suppress ffmpeg logs
        )
        .overwrite_output()  # Allow overwriting the output file if it exists
        .run()  # Execute the command
    )

    # Rename temp video to final name (muxing is unnecessary - TalkNet reads audio/video separately)
    os.rename(cropFile + "t.avi", cropFile + ".avi")

    return {"track": track, "proc_track": dets}
