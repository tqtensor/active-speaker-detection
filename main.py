import glob
import math
import os
import pickle
import subprocess
import warnings
from shutil import rmtree

import numpy
import torch
import tqdm

from config.args import get_args
from config.logging_config import get_logger, log_section, setup_logging
from model.talkNet import talkNet
from model.yoloFace import run_face_detection
from utils.helpers import export_metadata, summarize_tracks, visualization
from utils.inference_utils import get_speaker_track_indices
from utils.track_utils import scene_detect, track_shot
from utils.video_utils import extract_audio, extract_video

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

# YOLO variant download URLs from YapaLab/yolo-face releases
YOLO_FACE_URLS = {
    "n": "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11n-face.pt",
    "s": "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11s-face.pt",
    "m": "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11m-face.pt",
    "l": "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11l-face.pt",
}

LIGHTASD_WEIGHT_URL = (
    "https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/main/"
    "weight/pretrain_AVA_CVPR.model"
)


def download_weights(args):
    """Downloads model weights if not present.

    Downloads TalkNet or Light-ASD (whichever is selected via args.asdModel)
    and YOLO face detection weights from their respective sources. Validates
    existing YOLO weights by attempting to load them with PyTorch to detect
    corruption.

    Args:
        args: Pipeline arguments containing asdModel, talkNetWeights,
            lightAsdWeights, yoloFaceWeights, and yoloVariant paths.

    Raises:
        RuntimeError: If YOLO or Light-ASD weights download fails or
            downloaded weights are corrupted.
    """
    if args.asdModel == "talknet":
        if not os.path.isfile(args.talkNetWeights):
            os.makedirs(os.path.dirname(args.talkNetWeights), exist_ok=True)
            subprocess.run(
                [
                    "gdown",
                    "--id",
                    "1OEZiw1mM5Au_46ylcdDBOhClqjY7sH3V",
                    "-O",
                    args.talkNetWeights,
                ],
                stdout=subprocess.DEVNULL,
            )

    if args.asdModel == "lightasd" and not os.path.isfile(args.lightAsdWeights):
        os.makedirs(os.path.dirname(args.lightAsdWeights), exist_ok=True)
        logger.info(f"Downloading Light-ASD weights from {LIGHTASD_WEIGHT_URL}...")
        result = subprocess.run(
            [
                "wget",
                "-q",
                "--show-progress",
                "-L",
                LIGHTASD_WEIGHT_URL,
                "-O",
                args.lightAsdWeights,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download Light-ASD weights: {result.stderr}")
        torch.load(args.lightAsdWeights, map_location="cpu")  # validate

    # Check if YOLO weights exist and are valid
    need_download = False
    if not os.path.isfile(args.yoloFaceWeights):
        need_download = True
    else:
        # Validate existing file by checking if it can be loaded
        try:
            # PyTorch 2.6+ requires weights_only=False for YOLO models
            # Safe because weights come from official ultralytics/YOLO repository
            torch.load(args.yoloFaceWeights, map_location="cpu", weights_only=False)
            logger.info(f"YOLO weights validated: {args.yoloFaceWeights}")
        except Exception as e:
            logger.warning(f"Existing YOLO weights corrupted ({e}), re-downloading...")
            need_download = True
            os.remove(args.yoloFaceWeights)

    if need_download:
        os.makedirs(os.path.dirname(args.yoloFaceWeights), exist_ok=True)
        yolo_url = YOLO_FACE_URLS[args.yoloVariant]
        logger.info(f"Downloading YOLO weights from {yolo_url}...")
        result = subprocess.run(
            [
                "wget",
                "-q",
                "--show-progress",
                "-L",
                yolo_url,
                "-O",
                args.yoloFaceWeights,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download YOLO weights: {result.stderr}")

        # Verify downloaded file
        try:
            # PyTorch 2.6+ requires weights_only=False for YOLO models
            # Safe because weights come from official ultralytics/YOLO repository
            torch.load(args.yoloFaceWeights, map_location="cpu", weights_only=False)
            logger.info("YOLO weights downloaded and validated successfully")
        except Exception as e:
            os.remove(args.yoloFaceWeights)
            raise RuntimeError(f"Downloaded YOLO weights are corrupted: {e}")


def prepare_paths(args):
    """Prepares output directories for the pipeline.

    Locates the video file, creates the output directory structure, and
    removes any existing output from previous runs to ensure clean state.

    Args:
        args: Pipeline arguments. Updated in-place with videoPath, savePath,
            pyaviPath, and pyworkPath attributes.
    """
    args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + ".*"))[0]
    args.savePath = os.path.join(args.videoFolder, args.videoName)
    args.pyaviPath = os.path.join(args.savePath, "pyavi")
    args.pyworkPath = os.path.join(args.savePath, "pywork")

    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.pyaviPath)
    os.makedirs(args.pyworkPath)


def run_asd_inference_gpu(allTracks, args):
    """Runs ASD inference with GPU-optimized face cropping.

    Processes video in memory-aware chunks to avoid GPU OOM errors. Decodes
    video frames directly on GPU, crops faces in batches, then runs batched
    TalkNet inference across all tracks and temporal segments.

    Args:
        allTracks: List of face track dicts, each containing 'frame' and 'bbox'
            arrays.
        args: Pipeline arguments containing videoFilePath, cropScale,
            talkNetWeights, and talknetBatchSize.

    Returns:
        Tuple of (vidTracks, scores) where vidTracks is a list of processed
        track metadata and scores is a list of numpy arrays containing per-frame
        active speaker scores for each track.
    """
    from utils.audio_features import (
        compute_global_mfcc,
        compute_proc_track,
        slice_track_mfcc,
    )
    from utils.gpu_crop import crop_faces_gpu
    from utils.gpu_video import (
        decode_video_chunked,
        get_video_info,
    )

    # One MFCC for the whole clip; sliced per track below (no per-track wav I/O).
    global_mfcc = compute_global_mfcc(args.audioFilePath)

    # proc_track (smoothed centers/size) needed by visualization/metadata.
    vidTracks = [
        {"track": track, "proc_track": compute_proc_track(track)} for track in allTracks
    ]

    logger.info("[GPU Mode] Chunked video processing...")
    video_info = get_video_info(args.videoFilePath)
    total_frames = video_info["num_frames"]

    # Clear GPU cache from previous stages (YOLO, etc.)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Calculate optimal chunk size based on FREE GPU memory (not total)
    # Account for: uint8 frames + float32 conversion (4x) + cropping overhead
    # Use 10% of free memory to be safe
    free_memory_mb, total_memory_mb = torch.cuda.mem_get_info(0)
    free_memory_mb = free_memory_mb / (1024 * 1024)
    total_memory_mb = total_memory_mb / (1024 * 1024)

    # Frame size: uint8 (3 bytes) + float32 conversion (12 bytes) = 15 bytes/pixel
    # Plus cropping/processing overhead, use ~20 bytes/pixel as estimate
    frame_size_mb = (video_info["height"] * video_info["width"] * 20) / (1024 * 1024)
    max_frames_in_gpu = int((free_memory_mb * 0.1) / frame_size_mb)
    chunk_size = max(50, min(max_frames_in_gpu, 500))  # Between 50-500 frames

    num_chunks = (total_frames + chunk_size - 1) // chunk_size
    logger.info(
        f"Video: {total_frames} frames, {num_chunks} chunks of {chunk_size} frames"
    )
    logger.debug(
        f"GPU memory: {free_memory_mb:.0f} MB free / {total_memory_mb:.0f} MB total, {frame_size_mb:.2f} MB/frame (with float32)"
    )

    # Process video in chunks, converting to TalkNet features immediately
    # to avoid accumulating full-resolution RGB float32 crops in RAM (~12x smaller)
    all_talknet_chunks = {}  # {track_idx: list of (T, 112, 112) tensors}

    for chunk_idx, start_frame, frames in tqdm.tqdm(
        decode_video_chunked(args.videoFilePath, chunk_size=chunk_size, device_id=0),
        total=num_chunks,
        desc="Decoding chunks",
    ):
        end_frame = start_frame + len(frames)

        # Find tracks that have frames in this chunk
        chunk_tracks = []
        chunk_track_indices = []
        for track_idx, track in enumerate(allTracks):
            track_frames = numpy.array(track["frame"])
            track_bboxes = numpy.array(track["bbox"])
            # Check if any frames from this track are in the current chunk
            mask = (track_frames >= start_frame) & (track_frames < end_frame)
            if mask.any():
                # Create a sub-track with only frames in this chunk
                sub_track = {
                    "frame": track_frames[mask]
                    - start_frame,  # Adjust to chunk-relative
                    "bbox": track_bboxes[mask],
                }
                chunk_tracks.append(sub_track)
                chunk_track_indices.append(track_idx)

        if chunk_tracks:
            # Crop faces for this chunk
            chunk_cropped = crop_faces_gpu(
                frames, chunk_tracks, crop_size=224, crop_scale=args.cropScale
            )

            # Convert to grayscale 112x112 TalkNet features immediately
            # rather than storing full RGB 224x224 float32 crops
            crop_offset = (224 - 112) // 2
            for local_idx, track_idx in enumerate(chunk_track_indices):
                if local_idx in chunk_cropped:
                    crops = chunk_cropped[local_idx]
                    gray = (
                        0.299 * crops[:, 0] + 0.587 * crops[:, 1] + 0.114 * crops[:, 2]
                    )
                    gray = gray[
                        :,
                        crop_offset : crop_offset + 112,
                        crop_offset : crop_offset + 112,
                    ]
                    if track_idx not in all_talknet_chunks:
                        all_talknet_chunks[track_idx] = []
                    all_talknet_chunks[track_idx].append(gray.cpu())
            del chunk_cropped

        # Free GPU memory for this chunk
        del frames
        torch.cuda.empty_cache()

    # Combine TalkNet features from all chunks
    logger.info("[GPU Mode] Combining TalkNet features (on CPU)...")
    talknet_features = {}
    for track_idx, chunk_list in all_talknet_chunks.items():
        talknet_features[track_idx] = torch.cat(chunk_list, dim=0)

    del all_talknet_chunks

    logger.info(f"Prepared {len(talknet_features)} tracks (on CPU)")

    # Load the selected ASD model
    if args.asdModel == "lightasd":
        from model.lightASD import lightASD

        s = lightASD()
        s.loadParameters(args.lightAsdWeights)
        use_cross_attention = False
    else:
        s = talkNet()
        s.loadParameters(args.talkNetWeights)
        use_cross_attention = True
    s.eval()

    durationSet = [1, 2, 3, 4, 5, 6]
    weights = [3, 3, 2, 1, 1, 1]

    # Step 1: Prepare TalkNet features (combine GPU video features + sliced audio)
    logger.info("[GPU Mode] Preparing TalkNet features...")
    all_track_data = []
    for track_idx in range(len(allTracks)):
        if track_idx not in talknet_features:
            all_track_data.append(None)
            continue

        videoFeature = talknet_features[track_idx].numpy()  # Already on CPU

        # Slice the global MFCC to this track's frame range (no disk I/O)
        track_frames = numpy.array(allTracks[track_idx]["frame"])
        audioFeature = slice_track_mfcc(
            global_mfcc, int(track_frames[0]), int(track_frames[-1])
        )

        # Align lengths
        length = min(
            (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
            videoFeature.shape[0] / 25,
        )
        audioFeature = audioFeature[: int(round(length * 100)), :]
        videoFeature = videoFeature[: int(round(length * 25)), :, :]

        all_track_data.append(
            {
                "audio": audioFeature,
                "video": videoFeature,
                "length": length,
            }
        )

    # Step 2: Batched TalkNet inference across all tracks
    logger.info(
        f"[GPU Mode] Running batched TalkNet inference (batch_size={args.talknetBatchSize})..."
    )

    # Initialize score storage
    all_track_scores = {i: [] for i in range(len(allTracks))}

    for dur_idx, duration in enumerate(durationSet):
        # Collect all segments across all tracks for this duration
        segments = []  # [(track_idx, seg_idx, audio_seg, video_seg), ...]

        for track_idx, track_data in enumerate(all_track_data):
            if track_data is None:
                continue

            length = track_data["length"]
            num_segs = int(math.ceil(length / duration))

            for seg_idx in range(num_segs):
                audio_seg = track_data["audio"][
                    seg_idx * duration * 100 : (seg_idx + 1) * duration * 100, :
                ]
                video_seg = track_data["video"][
                    seg_idx * duration * 25 : (seg_idx + 1) * duration * 25, :, :
                ]

                if audio_seg.shape[0] > 0 and video_seg.shape[0] > 0:
                    segments.append((track_idx, seg_idx, audio_seg, video_seg))

        # Process segments in batches
        segment_scores = {}  # {(track_idx, seg_idx): score}
        batch_size = args.talknetBatchSize

        with torch.no_grad():
            for batch_start in range(0, len(segments), batch_size):
                batch = segments[batch_start : batch_start + batch_size]

                # Pad to same length within batch
                max_audio_len = max(seg[2].shape[0] for seg in batch)
                max_video_len = max(seg[3].shape[0] for seg in batch)

                audio_batch = []
                video_batch = []
                for _, _, audio_seg, video_seg in batch:
                    # Pad audio
                    audio_padded = numpy.zeros((max_audio_len, 13), dtype=numpy.float32)
                    audio_padded[: audio_seg.shape[0], :] = audio_seg
                    audio_batch.append(audio_padded)

                    # Pad video
                    video_padded = numpy.zeros(
                        (max_video_len, 112, 112), dtype=numpy.float32
                    )
                    video_padded[: video_seg.shape[0], :, :] = video_seg
                    video_batch.append(video_padded)

                inputA = torch.FloatTensor(numpy.stack(audio_batch)).cuda()
                inputV = torch.FloatTensor(numpy.stack(video_batch)).cuda()

                # Forward pass
                embedA = s.model.forward_audio_frontend(inputA)
                embedV = s.model.forward_visual_frontend(inputV)
                if use_cross_attention:
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                out = s.model.forward_audio_visual_backend(embedA, embedV)
                scores = s.lossAV.forward(out, labels=None)

                # Normalize scores to a per-sample list of per-frame arrays.
                scores_np = numpy.asarray(scores).reshape(len(batch), -1)
                for i, (track_idx, seg_idx, audio_seg, _) in enumerate(batch):
                    # Trim score to actual segment length (not padded length)
                    actual_frames = (
                        audio_seg.shape[0] // 4
                    )  # MFCC frames to score frames
                    seg_score = scores_np[i][:actual_frames].tolist()
                    segment_scores[(track_idx, seg_idx)] = seg_score

        # Aggregate scores for each track
        for track_idx, track_data in enumerate(all_track_data):
            if track_data is None:
                continue

            length = track_data["length"]
            num_segs = int(math.ceil(length / duration))

            track_scores = []
            for seg_idx in range(num_segs):
                if (track_idx, seg_idx) in segment_scores:
                    seg_score = segment_scores[(track_idx, seg_idx)]
                    if isinstance(seg_score, list):
                        track_scores.extend(seg_score)
                    else:
                        track_scores.append(seg_score)

            # Apply weight for this duration
            for _ in range(weights[dur_idx]):
                all_track_scores[track_idx].append(track_scores)

    # Step 3: Calculate final scores for each track
    all_scores = []
    for track_idx in range(len(allTracks)):
        if all_track_data[track_idx] is None:
            all_scores.append(numpy.array([]))
        else:
            track_scores = all_track_scores[track_idx]
            if track_scores and any(len(s) > 0 for s in track_scores):
                # Find min length and align
                min_len = min(len(s) for s in track_scores if len(s) > 0)
                aligned_scores = [s[:min_len] for s in track_scores if len(s) > 0]
                final_score = numpy.round(
                    numpy.mean(numpy.array(aligned_scores), axis=0), 1
                ).astype(float)
                all_scores.append(final_score)
            else:
                all_scores.append(numpy.array([]))

    return vidTracks, all_scores


def main():
    """Runs the active speaker detection pipeline.

    Executes the full pipeline including video preprocessing, face detection,
    face tracking, GPU-optimized ASD inference (TalkNet or Light-ASD), and
    output generation.
    """
    args = get_args()

    # Initialize logging
    setup_logging(args)

    # Set yoloFaceWeights based on variant if not explicitly provided
    if args.yoloFaceWeights is None:
        args.yoloFaceWeights = f"./weights/yolo/yolov11{args.yoloVariant}-face.pt"

    download_weights(args)
    prepare_paths(args)

    log_section(logger, "ACTIVE SPEAKER DETECTION PIPELINE")

    # Clear any residual GPU memory from previous runs
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            logger.warning(f"Could not clear GPU cache ({e})")
            logger.warning("Try running: pkill -9 python && nvidia-smi")

    # Step 1: Preprocess
    logger.info("[1/5] Preprocessing video...")
    extract_video(args)
    extract_audio(args)

    # Step 2: Face detection
    logger.info("[2/5] Detecting faces...")
    scene = scene_detect(args)
    faces = run_face_detection(args, batch_size=args.yoloBatchSize)

    # Step 3: Face tracking
    logger.info("[3/5] Tracking faces...")
    allTracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            allTracks.extend(
                track_shot(args, faces[shot[0].frame_num : shot[1].frame_num])
            )
    logger.info(f"Found {len(allTracks)} face tracks")

    if not allTracks:
        logger.error("No face tracks found!")
        return

    # Clean up GPU memory from YOLO before TalkNet
    logger.info("Clearing GPU memory from face detection...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Step 4: ASD inference (GPU-optimized, model chosen via --asdModel)
    logger.info(f"[4/5] Running {args.asdModel} inference...")
    vidTracks, scores = run_asd_inference_gpu(allTracks, args)

    # Save results
    with open(os.path.join(args.pyworkPath, "tracks.pckl"), "wb") as f:
        pickle.dump(vidTracks, f)
    with open(os.path.join(args.pyworkPath, "scores.pckl"), "wb") as f:
        pickle.dump(scores, f)

    # Step 5: Output
    logger.info("[5/5] Generating output...")
    speaker_track_indices = get_speaker_track_indices(scores, args)

    if not args.metadataOnly:
        visualization(vidTracks, scores, args, speaker_track_indices)

    summarize_tracks(vidTracks, scores, args, speaker_track_indices)
    export_metadata(vidTracks, scores, args, speaker_track_indices)

    log_section(logger, "PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
