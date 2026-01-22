from typing import Dict, List

import torch
import torch.nn.functional as F


def crop_faces_gpu(
    frames: torch.Tensor,
    tracks: List[Dict],
    crop_size: int = 224,
    crop_scale: float = 0.4,
) -> Dict[int, torch.Tensor]:
    """Crops faces directly on GPU without disk I/O.

    Extracts face regions from frames based on tracked bounding boxes and
    resizes to uniform size for TalkNet input.

    Args:
        frames: Video frames on GPU, shape (N, H, W, 3), uint8.
        tracks: List of track dicts, each with 'frame' (array of frame indices)
            and 'bbox' (array of [x1, y1, x2, y2] for each frame).
        crop_size: Output face crop size (square).
        crop_scale: Padding scale for bounding box expansion.

    Returns:
        Dict mapping track_idx to cropped face tensors, each of shape
        (T, 3, crop_size, crop_size).
    """
    # Convert to float and normalize, reorder to (N, C, H, W)
    frames_float = frames.float() / 255.0
    frames_chw = frames_float.permute(0, 3, 1, 2)  # (N, 3, H, W)

    _, _, H, W = frames_chw.shape
    cropped_tracks = {}

    for track_idx, track in enumerate(tracks):
        frame_indices = track["frame"]
        bboxes = track["bbox"]

        crops = []
        for i, fidx in enumerate(frame_indices):
            if fidx >= len(frames_chw):
                continue

            frame = frames_chw[fidx]  # (3, H, W)
            x1, y1, x2, y2 = bboxes[i]

            # Calculate crop with padding
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            size = max(w, h) * (1 + crop_scale)

            # Crop coordinates (clamp to frame bounds)
            crop_x1 = int(max(0, cx - size / 2))
            crop_y1 = int(max(0, cy - size / 2))
            crop_x2 = int(min(W, cx + size / 2))
            crop_y2 = int(min(H, cy + size / 2))

            # Skip if crop is invalid
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            # Extract and resize
            face = frame[:, crop_y1:crop_y2, crop_x1:crop_x2].unsqueeze(0)
            face = F.interpolate(
                face, size=(crop_size, crop_size), mode="bilinear", align_corners=False
            )
            crops.append(face.squeeze(0))

        if crops:
            cropped_tracks[track_idx] = torch.stack(crops)  # (T, 3, 224, 224)

    return cropped_tracks


def extract_talknet_features_gpu(cropped_faces: torch.Tensor) -> torch.Tensor:
    """Converts cropped faces to TalkNet input format on GPU.

    TalkNet expects grayscale 112x112 center-cropped face images.

    Args:
        cropped_faces: Tensor of shape (T, 3, 224, 224) on GPU.

    Returns:
        Tensor of shape (T, 112, 112), grayscale and center-cropped.
    """
    # Convert RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
    # Input is (T, 3, H, W)
    gray = (
        0.299 * cropped_faces[:, 0]
        + 0.587 * cropped_faces[:, 1]
        + 0.114 * cropped_faces[:, 2]
    )

    # Center crop from 224 to 112
    start = (224 - 112) // 2
    gray = gray[:, start : start + 112, start : start + 112]

    return gray


def batch_crop_faces_gpu(
    frames: torch.Tensor,
    tracks: List[Dict],
    crop_size: int = 224,
    crop_scale: float = 0.4,
    batch_size: int = 100,
) -> Dict[int, torch.Tensor]:
    """Crops faces in batches for memory efficiency.

    Processes crops in batches to avoid GPU memory overflow for videos with
    many tracked faces.

    Args:
        frames: Video frames on GPU, shape (N, H, W, 3), uint8.
        tracks: List of track dicts.
        crop_size: Output face crop size.
        crop_scale: Bounding box padding scale.
        batch_size: Number of crops to process at once.

    Returns:
        Dict mapping track_idx to cropped face tensors.
    """
    # For small number of tracks, use simple version
    if len(tracks) <= batch_size:
        return crop_faces_gpu(frames, tracks, crop_size, crop_scale)

    # Process in batches for memory efficiency
    all_results = {}
    for batch_start in range(0, len(tracks), batch_size):
        batch_end = min(batch_start + batch_size, len(tracks))
        batch_tracks = tracks[batch_start:batch_end]

        batch_results = crop_faces_gpu(frames, batch_tracks, crop_size, crop_scale)

        # Remap indices to original track indices
        for local_idx, crops in batch_results.items():
            original_idx = batch_start + local_idx
            all_results[original_idx] = crops

    return all_results


def prepare_talknet_input_gpu(
    cropped_tracks: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    """Prepares all cropped tracks for TalkNet input.

    Converts all cropped RGB faces to grayscale 112x112 format expected by
    TalkNet.

    Args:
        cropped_tracks: Dict of {track_idx: (T, 3, 224, 224) tensor}.

    Returns:
        Dict of {track_idx: (T, 112, 112) tensor} ready for TalkNet.
    """
    return {
        idx: extract_talknet_features_gpu(crops)
        for idx, crops in cropped_tracks.items()
    }
