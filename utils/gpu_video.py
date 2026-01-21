"""GPU video decoding utilities using decord.

This module provides GPU-accelerated video decoding using decord with NVDEC
when available. Falls back to CPU decoding with GPU transfer if NVDEC is
not supported.
"""

from typing import Optional, Union

import numpy as np
import torch

try:
    import decord
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
    # Set PyTorch bridge for direct tensor output
    decord.bridge.set_bridge('torch')
except ImportError:
    DECORD_AVAILABLE = False


def decode_video_gpu(
    video_path: str,
    device_id: int = 0
) -> torch.Tensor:
    """Decode video directly to GPU memory using NVDEC if available.

    Attempts GPU decoding first (requires CUDA-enabled decord build).
    Falls back to CPU decoding with GPU transfer if GPU decode fails.

    Args:
        video_path: Path to video file.
        device_id: GPU device ID for CUDA operations.

    Returns:
        torch.Tensor: Frames on GPU, shape (N, H, W, 3), dtype uint8.

    Raises:
        ImportError: If decord is not installed.
    """
    if not DECORD_AVAILABLE:
        raise ImportError("decord is required for GPU video decoding. Install with: pip install decord")

    try:
        # Try GPU decoding first (requires CUDA-enabled decord)
        from decord import gpu
        vr = VideoReader(video_path, ctx=gpu(device_id))
        gpu_decode = True
    except Exception:
        # Fall back to CPU decoding
        vr = VideoReader(video_path, ctx=cpu(0))
        gpu_decode = False

    # Batch fetch all frames
    frame_indices = list(range(len(vr)))
    frames = vr.get_batch(frame_indices)

    # Ensure on GPU
    if not frames.is_cuda:
        frames = frames.cuda(device_id)

    return frames


def decode_video_cpu_to_gpu(
    video_path: str,
    device_id: int = 0
) -> torch.Tensor:
    """Decode on CPU, transfer to GPU (fallback method).

    Use this when GPU decoding is not available or for compatibility.

    Args:
        video_path: Path to video file.
        device_id: GPU device ID for CUDA transfer.

    Returns:
        torch.Tensor: Frames on GPU, shape (N, H, W, 3), dtype uint8.
    """
    if not DECORD_AVAILABLE:
        raise ImportError("decord is required. Install with: pip install decord")

    vr = VideoReader(video_path, ctx=cpu(0))
    frames = vr.get_batch(list(range(len(vr))))
    return frames.cuda(device_id)


def decode_video_chunked(
    video_path: str,
    chunk_size: int = 7500,
    device_id: int = 0
) -> torch.Tensor:
    """Decode video in chunks for memory-efficient processing.

    Useful for long videos that may exceed GPU memory if loaded entirely.
    Processes chunk_size frames at a time (default ~5 min at 25fps).

    Args:
        video_path: Path to video file.
        chunk_size: Number of frames per chunk (7500 = ~5 min at 25fps).
        device_id: GPU device ID.

    Yields:
        tuple: (chunk_idx, start_frame, torch.Tensor of frames on GPU)
    """
    if not DECORD_AVAILABLE:
        raise ImportError("decord is required. Install with: pip install decord")

    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    for chunk_idx, start in enumerate(range(0, total_frames, chunk_size)):
        end = min(start + chunk_size, total_frames)
        frame_indices = list(range(start, end))
        frames = vr.get_batch(frame_indices)

        if not frames.is_cuda:
            frames = frames.cuda(device_id)

        yield chunk_idx, start, frames


def get_video_info(video_path: str) -> dict:
    """Get video metadata without loading all frames.

    Args:
        video_path: Path to video file.

    Returns:
        dict: Video info including num_frames, fps, width, height.
    """
    if not DECORD_AVAILABLE:
        raise ImportError("decord is required. Install with: pip install decord")

    vr = VideoReader(video_path, ctx=cpu(0))
    return {
        'num_frames': len(vr),
        'fps': vr.get_avg_fps(),
        'width': vr[0].shape[1],
        'height': vr[0].shape[0],
    }


def estimate_gpu_memory_mb(num_frames: int, height: int = 720, width: int = 1280) -> float:
    """Estimate GPU memory needed for video frames.

    Args:
        num_frames: Number of frames.
        height: Frame height in pixels.
        width: Frame width in pixels.

    Returns:
        float: Estimated memory in megabytes.
    """
    # RGB uint8: 3 bytes per pixel
    bytes_per_frame = height * width * 3
    total_bytes = num_frames * bytes_per_frame
    return total_bytes / (1024 * 1024)
