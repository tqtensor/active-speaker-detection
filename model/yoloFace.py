import glob
import os
import pickle
import sys

import cv2
from ultralytics import YOLO


def load_yolo_model(weights_path):
    """Loads a YOLO model from the specified weights file.

    Args:
        weights_path: Path to the YOLO model weights file.

    Returns:
        Initialized YOLO model instance.
    """
    return YOLO(weights_path)


def run_face_detection_single(model, frames_path, video_file_path, work_path):
    """Runs face detection on individual frames using a YOLO model.

    Processes frames sequentially from the specified directory, detects faces
    with confidence threshold of 0.7, and saves results as a pickle file.

    Args:
        model: YOLO model instance for face detection.
        frames_path: Directory path containing frame images (.jpg files).
        video_file_path: Video file path used for progress reporting.
        work_path: Working directory where results will be saved.

    Returns:
        List of detection results, where each element is a list of face
        detections for the corresponding frame. Each detection contains
        frame index, bounding box coordinates, and confidence score.
    """
    flist = sorted(glob.glob(os.path.join(frames_path, "*.jpg")))
    dets = []

    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        results = model.predict(image, conf=0.7, iou=0.5)
        dets.append(
            [
                {
                    "frame": fidx,
                    "bbox": box.xyxy.cpu().numpy().tolist()[0],
                    "conf": float(box.conf.item()),
                }
                for box in results[0].boxes
            ]
        )
        sys.stderr.write("%s-%05d; %d dets\r" % (video_file_path, fidx, len(dets[-1])))

    with open(os.path.join(work_path, "faces.pckl"), "wb") as fil:
        pickle.dump(dets, fil)

    return dets


def run_face_detection(args, batch_size=32):
    """Runs batch face detection on video frames using YOLO.

    Processes frames in batches for improved performance, detects faces with
    confidence threshold of 0.7, and saves results to the working directory.

    Args:
        args: Configuration object containing yoloFaceWeights (path to YOLO
            weights), pyframesPath (directory with extracted frames), and
            pyworkPath (working directory for output).
        batch_size: Number of frames to process in each batch.

    Returns:
        List of detection results, where each element is a list of face
        detections for the corresponding frame. Each detection contains
        frame index, bounding box coordinates, and confidence score.
    """
    model = YOLO(args.yoloFaceWeights)

    flist = sorted(glob.glob(os.path.join(args.pyframesPath, "*.jpg")))
    dets = [None] * len(flist)  # Pre-allocate to maintain frame order

    # Process frames in batches
    for batch_start in range(0, len(flist), batch_size):
        batch_end = min(batch_start + batch_size, len(flist))
        batch_files = flist[batch_start:batch_end]

        # Load batch of images
        batch_images = [cv2.imread(fname) for fname in batch_files]

        # Batch inference
        results = model.predict(batch_images, conf=0.7, iou=0.5, verbose=False)

        # Process results for each image in batch
        for i, result in enumerate(results):
            fidx = batch_start + i
            dets[fidx] = [
                {
                    "frame": fidx,
                    "bbox": box.xyxy.cpu().numpy().tolist()[0],
                    "conf": float(box.conf.item()),
                }
                for box in result.boxes
            ]

        sys.stderr.write("Face detection: %05d/%05d frames\r" % (batch_end, len(flist)))

    sys.stderr.write("\n")

    with open(os.path.join(args.pyworkPath, "faces.pckl"), "wb") as fil:
        pickle.dump(dets, fil)

    return dets
