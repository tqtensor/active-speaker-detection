

from typing import Tuple

import av
import numpy as np
import torch


def decode_video_gpu(video_path: str, device_id: int = 0) -> Tuple[torch.Tensor, bool]:
    """Decodes video directly to GPU memory using NVDEC if available.

    Attempts GPU decoding first (requires NVDEC-capable GPU).
    Falls back to CPU decoding with GPU transfer if GPU decode fails.

    Args:
        video_path: Path to video file.
        device_id: GPU device ID for CUDA operations.

    Returns:
        Tuple containing frames tensor on GPU with shape (N, H, W, 3) and
        dtype uint8, and boolean indicating if NVDEC was used.

    """
    gpu_decode = False
    frames_list = []

    try:
        # Try GPU decoding with NVDEC
        container = av.open(
            video_path,
            options={
                "hwaccel": "cuda",
                "hwaccel_device": str(device_id),
            },
        )
        gpu_decode = True
        print(f"  ✓ Using NVDEC hardware acceleration on GPU {device_id}")
    except Exception as e:
        # Fall back to CPU decoding
        print(f"  ⚠ NVDEC not available ({type(e).__name__}), using CPU decode")
        container = av.open(video_path)

    # Decode all frames
    for frame in container.decode(video=0):
        # Convert to numpy array (RGB24 format)
        img = frame.to_ndarray(format="rgb24")
        frames_list.append(img)

    container.close()

    # Stack frames and convert to tensor
    frames_np = np.stack(frames_list)
    frames_tensor = torch.from_numpy(frames_np)

    # Move to GPU if not already there
    if not frames_tensor.is_cuda:
        frames_tensor = frames_tensor.cuda(device_id)

    return frames_tensor, gpu_decode


def decode_video_cpu_to_gpu(video_path: str, device_id: int = 0) -> torch.Tensor:
    """Decodes video on CPU and transfers frames to GPU memory.

    Use this when you want to force CPU decoding, or for compatibility.

    Args:
        video_path: Path to video file.
        device_id: GPU device ID for CUDA transfer.

    Returns:
        Frames on GPU with shape (N, H, W, 3) and dtype uint8.
    """
    frames_list = []
    container = av.open(video_path)

    for frame in container.decode(video=0):
        img = frame.to_ndarray(format="rgb24")
        frames_list.append(img)

    container.close()

    frames_np = np.stack(frames_list)
    frames_tensor = torch.from_numpy(frames_np)
    return frames_tensor.cuda(device_id)


def decode_video_chunked(video_path: str, chunk_size: int = 7500, device_id: int = 0):
    """Decodes video in chunks for memory-efficient processing.

    Useful for long videos that may exceed GPU memory if loaded entirely.
    Processes chunk_size frames at a time (default ~5 min at 25fps).

    Args:
        video_path: Path to video file.
        chunk_size: Number of frames per chunk (7500 = ~5 min at 25fps).
        device_id: GPU device ID.

    Yields:
        Tuple containing chunk index, start frame number, and tensor of
        frames on GPU.
    """
    container = av.open(video_path)

    chunk_idx = 0
    start_frame = 0
    frames_list = []

    for frame_idx, frame in enumerate(container.decode(video=0)):
        img = frame.to_ndarray(format="rgb24")
        frames_list.append(img)

        # Yield chunk when full
        if len(frames_list) >= chunk_size:
            frames_np = np.stack(frames_list)
            frames_tensor = torch.from_numpy(frames_np).cuda(device_id)
            yield chunk_idx, start_frame, frames_tensor

            chunk_idx += 1
            start_frame = frame_idx + 1
            frames_list = []

    # Yield remaining frames
    if frames_list:
        frames_np = np.stack(frames_list)
        frames_tensor = torch.from_numpy(frames_np).cuda(device_id)
        yield chunk_idx, start_frame, frames_tensor

    container.close()


def get_video_info(video_path: str) -> dict:
    """Retrieves video metadata without loading all frames.

    Args:
        video_path: Path to video file.

    Returns:
        Dictionary containing num_frames, fps, width, and height.
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    info = {
        "num_frames": video_stream.frames,
        "fps": float(video_stream.average_rate),
        "width": video_stream.width,
        "height": video_stream.height,
    }

    container.close()
    return info


def estimate_gpu_memory_mb(
    num_frames: int, height: int = 720, width: int = 1280
) -> float:
    """Estimates GPU memory needed for video frames.

    Args:
        num_frames: Number of frames.
        height: Frame height in pixels.
        width: Frame width in pixels.

    Returns:
        Estimated memory in megabytes.
    """
    # RGB uint8: 3 bytes per pixel
    bytes_per_frame = height * width * 3
    total_bytes = num_frames * bytes_per_frame
    return total_bytes / (1024 * 1024)
