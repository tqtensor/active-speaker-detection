import os
import pickle

import numpy
from ultralytics import YOLO

from config.logging_config import get_logger
from utils.gpu_video import decode_video_chunked, get_video_info

logger = get_logger(__name__)


def load_yolo_model(weights_path):
    """Loads a YOLO model from the specified weights file."""
    return YOLO(weights_path)


def run_face_detection(args, batch_size=32):
    """Runs batched face detection over in-memory decoded frames.

    Decodes ``args.videoFilePath`` in chunks (no intermediate JPEGs), runs
    YOLO in batches, and returns per-frame detections. Detections are computed
    on BGR frames to match the previous cv2-based behavior.

    Args:
        args: Config with videoFilePath and yoloFaceWeights.
        batch_size: Frames per YOLO forward pass.

    Returns:
        List indexed by frame; each element is a list of
        {"frame", "bbox", "conf"} dicts.
    """
    model = YOLO(args.yoloFaceWeights)

    num_frames = get_video_info(args.videoFilePath)["num_frames"]
    # PyAV container metadata for num_frames can be wrong (missing/low/0) for
    # some inputs. Accumulate detections keyed by the ACTUAL decoded frame
    # index rather than pre-sizing a list, so a metadata/decode mismatch
    # can't cause an IndexError mid-run.
    dets_by_frame = {}
    max_fidx = -1

    for _chunk_idx, start_frame, frames in decode_video_chunked(
        args.videoFilePath, chunk_size=max(batch_size, 256), device_id=0
    ):
        # (N,H,W,3) uint8 RGB on GPU -> CPU numpy, RGB->BGR to match cv2.imread.
        # ascontiguousarray removes the negative stride from the ::-1 reversal
        # so Ultralytics' torch.from_numpy preprocessing does not fail.
        chunk = numpy.ascontiguousarray(frames.cpu().numpy()[:, :, :, ::-1])
        del frames

        for b in range(0, len(chunk), batch_size):
            batch_imgs = [chunk[i] for i in range(b, min(b + batch_size, len(chunk)))]
            results = model.predict(batch_imgs, conf=0.7, iou=0.5, verbose=False)
            for i, result in enumerate(results):
                fidx = start_frame + b + i
                dets_by_frame[fidx] = [
                    {
                        "frame": fidx,
                        "bbox": box.xyxy.cpu().numpy().tolist()[0],
                        "conf": float(box.conf.item()),
                    }
                    for box in result.boxes
                ]
                max_fidx = max(max_fidx, fidx)

    # Build the ordered list, covering both the metadata frame count and any
    # actual decoded index beyond it. Any frame with no detections (either
    # skipped by the decoder, or beyond the metadata count) stays an empty list.
    total = max(num_frames, max_fidx + 1)
    dets = [dets_by_frame.get(i, []) for i in range(total)]

    with open(os.path.join(args.pyworkPath, "faces.pckl"), "wb") as fil:
        pickle.dump(dets, fil)

    return dets
