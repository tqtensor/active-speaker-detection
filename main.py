"""Optimized Active Speaker Detection Pipeline.

This pipeline provides GPU-accelerated active speaker detection with:
1. Decord-based fast video loading (2-3x faster than cv2.VideoCapture)
2. GPU video decoding when available
3. GPU face cropping (eliminates intermediate disk I/O)
4. Batched TalkNet inference option

Expected speedup: 4-10x faster than sequential processing.
"""

import glob
import math
import os
import pickle
import subprocess
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from shutil import rmtree

import numpy
import torch
import tqdm

from config.args import get_args
from model.talkNet import talkNet
from model.yoloFace import run_face_detection
from utils.helpers import export_metadata, summarize_tracks, visualization
from utils.inference_utils import get_speaker_track_indices
from utils.track_utils import crop_video, extract_audio_only, scene_detect, track_shot
from utils.video_utils import extract_audio, extract_frames, extract_video

warnings.filterwarnings("ignore")

# YOLO variant download URLs from YapaLab/yolo-face releases
YOLO_FACE_URLS = {
    "n": "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11n-face.pt",
    "s": "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11s-face.pt",
    "m": "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11m-face.pt",
    "l": "https://github.com/YapaLab/yolo-face/releases/download/1.0.0/yolov11l-face.pt",
}


def crop_video_worker(params):
    """Executes crop_video for parallel processing."""
    args, track, crop_path = params
    return crop_video(args, track, crop_path)


def extract_audio_worker(params):
    """Executes extract_audio_only for parallel processing (GPU mode)."""
    args, track, crop_path = params
    return extract_audio_only(args, track, crop_path)


def download_weights(args):
    """Download model weights if not present."""
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
            print(f"  YOLO weights validated: {args.yoloFaceWeights}")
        except Exception as e:
            print(f"  Existing YOLO weights corrupted ({e}), re-downloading...")
            need_download = True
            os.remove(args.yoloFaceWeights)

    if need_download:
        os.makedirs(os.path.dirname(args.yoloFaceWeights), exist_ok=True)
        yolo_url = YOLO_FACE_URLS[args.yoloVariant]
        print(f"  Downloading YOLO weights from {yolo_url}...")
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
            print("  YOLO weights downloaded and validated successfully")
        except Exception as e:
            os.remove(args.yoloFaceWeights)
            raise RuntimeError(f"Downloaded YOLO weights are corrupted: {e}")


def prepare_paths(args):
    """Prepare output directories."""
    args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + ".*"))[0]
    args.savePath = os.path.join(args.videoFolder, args.videoName)
    args.pyaviPath = os.path.join(args.savePath, "pyavi")
    args.pyframesPath = os.path.join(args.savePath, "pyframes")
    args.pyworkPath = os.path.join(args.savePath, "pywork")
    args.pycropPath = os.path.join(args.savePath, "pycrop")

    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.pyaviPath)
    os.makedirs(args.pyframesPath)
    os.makedirs(args.pyworkPath)
    os.makedirs(args.pycropPath)


def run_talknet_inference_gpu(allTracks, args):
    """Run TalkNet inference with GPU-optimized face cropping.

    Uses chunked GPU processing to handle videos of any length without OOM.
    Falls back to standard processing if GPU cropping is not available.
    """
    import python_speech_features
    from scipy.io import wavfile

    from utils.gpu_crop import crop_faces_gpu
    from utils.gpu_video import (
        decode_video_chunked,
        get_video_info,
    )

    print("\n[GPU Mode] Chunked video processing...")
    video_info = get_video_info(args.videoPath)
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
    print(f"  Video: {total_frames} frames, {num_chunks} chunks of {chunk_size} frames")
    print(
        f"  GPU memory: {free_memory_mb:.0f} MB free / {total_memory_mb:.0f} MB total, {frame_size_mb:.2f} MB/frame (with float32)"
    )

    # Process video in chunks, accumulating cropped faces
    all_cropped_tracks = {}  # {track_idx: list of (T, 3, 224, 224) tensors}

    for chunk_idx, start_frame, frames in tqdm.tqdm(
        decode_video_chunked(args.videoPath, chunk_size=chunk_size, device_id=0),
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

            # Store results mapped to original track indices
            for local_idx, track_idx in enumerate(chunk_track_indices):
                if local_idx in chunk_cropped:
                    if track_idx not in all_cropped_tracks:
                        all_cropped_tracks[track_idx] = []
                    all_cropped_tracks[track_idx].append(chunk_cropped[local_idx].cpu())

        # Free GPU memory for this chunk
        del frames
        torch.cuda.empty_cache()

    # Combine cropped faces from all chunks (KEEP ON CPU to avoid OOM)
    print("[GPU Mode] Combining cropped faces (on CPU)...")
    combined_cropped = {}
    for track_idx, crop_list in all_cropped_tracks.items():
        combined_cropped[track_idx] = torch.cat(crop_list, dim=0)  # Stay on CPU

    # Free the chunk lists
    del all_cropped_tracks
    torch.cuda.empty_cache()

    # Prepare TalkNet features (convert to grayscale 112x112) - on CPU
    talknet_features = {}
    for track_idx, crops in combined_cropped.items():
        # Convert RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        gray = 0.299 * crops[:, 0] + 0.587 * crops[:, 1] + 0.114 * crops[:, 2]
        # Center crop from 224 to 112
        start = (224 - 112) // 2
        talknet_features[track_idx] = gray[:, start : start + 112, start : start + 112]

    # Free the RGB crops
    del combined_cropped

    print(f"  Prepared {len(talknet_features)} tracks (on CPU)")

    # Load TalkNet model
    s = talkNet()
    s.loadParameters(args.talkNetWeights)
    s.eval()

    durationSet = [1, 2, 3, 4, 5, 6]
    weights = [3, 3, 2, 1, 1, 1]

    # Step 1: Extract audio for all tracks in parallel (skip video cropping - GPU already has it)
    print("[GPU Mode] Extracting audio in parallel...")
    num_workers = min(args.nDataLoaderThread, len(allTracks))
    audio_params = [
        (args, track, os.path.join(args.pycropPath, "%05d" % ii))
        for ii, track in enumerate(allTracks)
    ]

    vidTracks = [None] * len(allTracks)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(extract_audio_worker, params): idx
            for idx, params in enumerate(audio_params)
        }
        for future in tqdm.tqdm(
            as_completed(future_to_idx), total=len(allTracks), desc="Extracting audio"
        ):
            idx = future_to_idx[future]
            vidTracks[idx] = future.result()

    # Step 2: Prepare TalkNet features (combine GPU video features + extracted audio)
    print("[GPU Mode] Preparing TalkNet features...")
    all_track_data = []
    for track_idx in range(len(allTracks)):
        crop_path = os.path.join(args.pycropPath, "%05d" % track_idx)

        if track_idx not in talknet_features:
            all_track_data.append(None)
            continue

        videoFeature = talknet_features[track_idx].numpy()  # Already on CPU

        # Load audio features
        _, audio = wavfile.read(crop_path + ".wav")
        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010
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
    print(
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
                embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                out = s.model.forward_audio_visual_backend(embedA, embedV)
                scores = s.lossAV.forward(out, labels=None)

                # Store scores per segment
                for i, (track_idx, seg_idx, audio_seg, _) in enumerate(batch):
                    # Trim score to actual segment length (not padded length)
                    actual_frames = (
                        audio_seg.shape[0] // 4
                    )  # MFCC frames to score frames
                    seg_score = (
                        scores[i][:actual_frames]
                        if isinstance(scores[i], list)
                        else [scores[i]]
                    )
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


def run_talknet_inference_standard(allTracks, args):
    """Run TalkNet inference with standard processing (parallel video cropping)."""
    from utils.inference_utils import evaluate_network, evaluate_network_batched

    # Parallel crop_video processing
    num_workers = min(args.nDataLoaderThread, len(allTracks))
    crop_params = [
        (args, track, os.path.join(args.pycropPath, "%05d" % ii))
        for ii, track in enumerate(allTracks)
    ]

    vidTracks = [None] * len(allTracks)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(crop_video_worker, params): idx
            for idx, params in enumerate(crop_params)
        }
        for future in tqdm.tqdm(
            as_completed(future_to_idx), total=len(allTracks), desc="Cropping"
        ):
            idx = future_to_idx[future]
            vidTracks[idx] = future.result()

    # Run inference
    files = sorted(glob.glob(f"{args.pycropPath}/*.avi"))
    if args.useBatched:
        scores = evaluate_network_batched(files, args, batch_size=args.talknetBatchSize)
    else:
        scores = evaluate_network(files, args)

    return vidTracks, scores


def main():
    """Run the active speaker detection pipeline."""
    args = get_args()

    # Set yoloFaceWeights based on variant if not explicitly provided
    if args.yoloFaceWeights is None:
        args.yoloFaceWeights = f"./weights/yolo/yolov11{args.yoloVariant}-face.pt"

    download_weights(args)
    prepare_paths(args)

    print("=" * 60)
    print("ACTIVE SPEAKER DETECTION PIPELINE")
    print("=" * 60)

    # Clear any residual GPU memory from previous runs
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  Warning: Could not clear GPU cache ({e})")
            print("  Try running: pkill -9 python && nvidia-smi")

    # Step 1: Preprocess
    print("\n[1/5] Preprocessing video...")
    extract_video(args)
    extract_audio(args)
    extract_frames(args)

    # Step 2: Face detection
    print("\n[2/5] Detecting faces...")
    scene = scene_detect(args)
    faces = run_face_detection(args, batch_size=args.yoloBatchSize)

    # Step 3: Face tracking
    print("\n[3/5] Tracking faces...")
    allTracks = []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            allTracks.extend(
                track_shot(args, faces[shot[0].frame_num : shot[1].frame_num])
            )
    print(f"  Found {len(allTracks)} face tracks")

    if not allTracks:
        print("No face tracks found!")
        return

    # Clean up GPU memory from YOLO before TalkNet
    print("\n  Clearing GPU memory from face detection...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Step 4: TalkNet inference (choose GPU or standard mode)
    print("\n[4/5] Running TalkNet inference...")
    try:
        # Try GPU-optimized path
        vidTracks, scores = run_talknet_inference_gpu(allTracks, args)
    except Exception as e:
        print(f"  GPU mode failed ({e}), falling back to standard mode...")
        vidTracks, scores = run_talknet_inference_standard(allTracks, args)

    # Save results
    with open(os.path.join(args.pyworkPath, "tracks.pckl"), "wb") as f:
        pickle.dump(vidTracks, f)
    with open(os.path.join(args.pyworkPath, "scores.pckl"), "wb") as f:
        pickle.dump(scores, f)

    # Step 5: Output
    print("\n[5/5] Generating output...")
    speaker_track_indices = get_speaker_track_indices(scores, args)

    if not args.metadataOnly:
        visualization(vidTracks, scores, args, speaker_track_indices)

    summarize_tracks(vidTracks, scores, args, speaker_track_indices)
    export_metadata(vidTracks, scores, args, speaker_track_indices)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
