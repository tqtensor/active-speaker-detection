import glob
import os
import pickle
import sys

import cv2
import ffmpeg
import numpy
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import SceneManager
from scenedetect.stats_manager import StatsManager
from scenedetect.video_manager import VideoManager
from scipy import signal
from scipy.interpolate import interp1d
from scipy.io import wavfile


def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    # print('args.videoFilePath', args.videoFilePath)
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    # print('baseTimecode', baseTimecode)
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source=videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(args.pyworkPath, "scene.pckl")
    if sceneList == []:
        sceneList = [
            (videoManager.get_base_timecode(), videoManager.get_current_timecode())
        ]
    with open(savePath, "wb") as fil:
        pickle.dump(sceneList, fil)
        sys.stderr.write(
            "%s - scenes detected %d\n" % (args.videoFilePath, len(sceneList))
        )
    return sceneList


def bb_intersection_over_union(boxA, boxB, evalCol=False):
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


def crop_video(args, track, cropFile):
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

    _, audio = wavfile.read(audioTmp)

    (
        ffmpeg.output(
            ffmpeg.input(f"{cropFile}t.avi"),  # Input video file
            ffmpeg.input(audioTmp),  # Input audio file
            f"{cropFile}.avi",  # Output file path
            c="copy",  # Copy codec (no re-encoding)
            threads=args.nDataLoaderThread,  # Use specified number of threads
            loglevel="panic",  # Suppress ffmpeg logs
        )
        .overwrite_output()
        .run()
    )
    os.remove(cropFile + "t.avi")
    return {"track": track, "proc_track": dets}
