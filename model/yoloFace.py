import os
import pickle

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
    dets = [None] * num_frames

    for _chunk_idx, start_frame, frames in decode_video_chunked(
        args.videoFilePath, chunk_size=max(batch_size, 256), device_id=0
    ):
        # (N,H,W,3) uint8 RGB on GPU -> CPU numpy, RGB->BGR to match cv2.imread
        chunk = frames.cpu().numpy()[:, :, :, ::-1]
        del frames

        for b in range(0, len(chunk), batch_size):
            batch_imgs = [chunk[i] for i in range(b, min(b + batch_size, len(chunk)))]
            results = model.predict(batch_imgs, conf=0.7, iou=0.5, verbose=False)
            for i, result in enumerate(results):
                fidx = start_frame + b + i
                dets[fidx] = [
                    {
                        "frame": fidx,
                        "bbox": box.xyxy.cpu().numpy().tolist()[0],
                        "conf": float(box.conf.item()),
                    }
                    for box in result.boxes
                ]

    # Any frame the decoder skipped stays an empty detection list.
    dets = [d if d is not None else [] for d in dets]

    with open(os.path.join(args.pyworkPath, "faces.pckl"), "wb") as fil:
        pickle.dump(dets, fil)

    return dets
