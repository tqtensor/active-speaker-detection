import glob
import json
import os
import sys
import time

import cv2
import ffmpeg
import numpy
import tqdm


def visualization(tracks, scores, args, speaker_track_indices):
    """Visualizes active speaker results with bounding boxes and speaking duration counters.

    Skips frames with multiple speakers if args.ignoreMultiSpeakers is enabled.

    Args:
        tracks: List of track dictionaries containing frame and bounding box data.
        scores: List of per-frame speaking scores per track.
        args: Arguments containing visualization parameters (pyframesPath, pyaviPath,
            speakerThresh, ignoreMultiSpeakers, minSpeechLen, nDataLoaderThread).
        speaker_track_indices: List of track indices that have been identified as speakers.
    """
    flist = sorted(glob.glob(os.path.join(args.pyframesPath, "*.jpg")))
    faces = [[] for _ in range(len(flist))]

    # Create a mapping from track indices to speaker IDs (tracks with speaking frames)
    speaker_tracks_id_map = {
        tidx: sid for sid, tidx in enumerate(speaker_track_indices)
    }
    speaker_duration_counter = {}

    # Iterate through each track and its corresponding score
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        frames = track["track"]["frame"].tolist()

        # Get the track ID for this track (set to -1 if not found)
        speaker_track_id = speaker_tracks_id_map.get(tidx, -1)
        # print('speaker_track_id', speaker_track_id)

        # Iterate through each frame in the track
        for fidx, frame in enumerate(frames):
            s = score[max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)]
            s = numpy.mean(s)

            # Determine if the track is speaking
            is_speaking = s >= args.speakerThresh

            # Update duration counter for the track if it is speaking
            if is_speaking and speaker_track_id != -1:
                speaker_duration_counter[speaker_track_id] = (
                    speaker_duration_counter.get(speaker_track_id, 0) + 1
                )
            else:
                speaker_duration_counter[speaker_track_id] = 0

            # Append face data
            faces[frame].append(
                {
                    "track": tidx,
                    "speaker_track_id": speaker_track_id,
                    "score": float(s),
                    "s": track["proc_track"]["s"][fidx],
                    "x": track["proc_track"]["x"][fidx],
                    "y": track["proc_track"]["y"][fidx],
                    "speaking": is_speaking,
                    "duration": speaker_duration_counter.get(speaker_track_id, 0)
                    / 25.0,  # seconds
                }
            )

    firstImage = cv2.imread(flist[0])
    fw, fh = firstImage.shape[1], firstImage.shape[0]
    vOut = cv2.VideoWriter(
        os.path.join(args.pyaviPath, "video_only.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        25,
        (fw, fh),
    )

    # Define color mapping for speaking and non-speaking
    colorDict = {0: 0, 1: 255}

    for fidx, fname in tqdm.tqdm(enumerate(flist), total=len(flist)):
        image = cv2.imread(fname)

        # Check if there are any speaking faces in the current frame
        speaking_faces = [face for face in faces[fidx] if face["speaking"]]

        # Skip frames with multiple speakers if ignoreMultiSpeakers is enabled
        multiple_speakers = (
            len(speaking_faces) > 1
        )  # check if more than one speaker is detected
        if args.ignoreMultiSpeakers and multiple_speakers:
            continue

        # Draw bounding boxes and labels for each face
        for face in faces[fidx]:
            # clr = colorDict[int(face['score'] >= args.speakerThresh)]
            if face["speaker_track_id"] != -1 and face["duration"] > args.minSpeechLen:
                clr = colorDict[1]
                txt = f"{round(face['score'], 2)}, ID:{face['speaker_track_id']}, Dur:{face['duration']:.2f}s"
            else:
                clr = colorDict[0]
                txt = f"{round(face['score'], 2)}"

            cv2.rectangle(
                image,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                (100, clr, 255 - clr),
                5,
            )

            cv2.putText(
                image,
                txt,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"] + 200)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, clr, 255 - clr),
                2,
            )

        vOut.write(image)

    vOut.release()

    # Add audio to the visualization (final output)
    (
        ffmpeg.output(
            ffmpeg.input(os.path.join(args.pyaviPath, "video_only.avi")),
            ffmpeg.input(os.path.join(args.pyaviPath, "audio.wav")),
            os.path.join(args.pyaviPath, "video_out.avi"),
            c="copy",
            threads=args.nDataLoaderThread,
            loglevel="panic",
        )
        .overwrite_output()
        .run()
    )

    return


def summarize_tracks(tracks, scores, args, speaker_track_indices):
    """Creates a JSON summary of speaking intervals for each speaker ID.

    Args:
        tracks: List of track dictionaries containing frame and bounding box data.
        scores: List of per-frame speaking scores per track.
        args: Arguments containing speakerThresh, minSpeechLen, videoName, and pyworkPath.
        speaker_track_indices: List of track indices that have been identified as speakers.

    Returns:
        Output dictionary containing video_name, fps, and speakers with their
        speaking intervals and total speaking time.
    """

    # Assign compact speaker IDs
    speaker_tracks_id_map = {
        tidx: sid for sid, tidx in enumerate(speaker_track_indices)
    }

    # Build speaking intervals only for speaker tracks
    speaking_intervals = {}

    for tidx, track in enumerate(tracks):
        if tidx not in speaker_tracks_id_map:
            continue  # skip non-speakers

        speaker_id = speaker_tracks_id_map[tidx]
        score = scores[tidx]
        frames = track["track"]["frame"].tolist()

        if speaker_id not in speaking_intervals:
            speaking_intervals[speaker_id] = []

        speaking = False
        start_frame = None

        for fidx, frame in enumerate(frames):
            s = score[max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)]
            s = numpy.mean(s)

            is_speaking = s >= args.speakerThresh

            if is_speaking and not speaking:
                speaking = True
                start_frame = frame

            elif (not is_speaking and speaking) or (
                is_speaking and speaking and fidx == len(frames) - 1
            ):
                speaking = False
                end_frame = frame
                start_time = start_frame / 25.0
                end_time = end_frame / 25.0
                duration = end_time - start_time

                # Only include intervals longer than args.minSpeechLen seconds
                if duration > args.minSpeechLen:
                    speaking_intervals[speaker_id].append(
                        {
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": duration,
                        }
                    )

    # Output JSON
    output = {"video_name": args.videoName, "fps": 25, "speakers": {}}

    for speaker_id, intervals in speaking_intervals.items():
        if intervals:
            output["speakers"][f"speaker_{speaker_id}"] = {
                "intervals": intervals,
                "total_speaking_time": sum(
                    interval["duration"] for interval in intervals
                ),
            }

    output_path = os.path.join(args.pyworkPath, "speaker_summary.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S")
        + f" Speaking intervals saved to {output_path}\n"
    )

    return output


def export_metadata(tracks, scores, args, speaker_track_indices):
    """Exports frame-centric metadata as human-readable JSON.

    Creates a comprehensive JSON file with per-frame face detection data,
    bounding boxes, speaking scores, and speaker identification.

    Args:
        tracks: List of track dictionaries containing frame and bounding box data.
        scores: List of per-frame speaking scores per track.
        args: Arguments containing pyframesPath, pyworkPath, videoName,
            speakerThresh, and minSpeechLen.
        speaker_track_indices: List of track indices that have been identified as speakers.

    Returns:
        Output dictionary containing the complete frame-centric metadata.
    """
    flist = sorted(glob.glob(os.path.join(args.pyframesPath, "*.jpg")))
    total_frames = len(flist)

    # Create a mapping from track indices to speaker IDs
    speaker_tracks_id_map = {
        tidx: sid for sid, tidx in enumerate(speaker_track_indices)
    }

    # Initialize frame data structure
    frames_data = [
        {"frame_id": i, "timestamp": i / 25.0, "faces": []} for i in range(total_frames)
    ]

    # Track duration counters for cumulative speaking time
    speaker_duration_counter = {}

    # Populate frame data from tracks
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        frame_indices = track["track"]["frame"].tolist()
        bboxes = track["track"]["bbox"]
        proc_track = track["proc_track"]

        speaker_id = speaker_tracks_id_map.get(tidx, None)

        for fidx, frame_num in enumerate(frame_indices):
            # Compute smoothed score (5-frame window)
            s = score[max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)]
            smoothed_score = float(numpy.mean(s))

            is_speaking = smoothed_score >= args.speakerThresh

            # Update duration counter
            if is_speaking and speaker_id is not None:
                speaker_duration_counter[speaker_id] = (
                    speaker_duration_counter.get(speaker_id, 0) + 1
                )
            elif speaker_id is not None:
                speaker_duration_counter[speaker_id] = 0

            # Get bounding box coordinates
            bbox = bboxes[fidx]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Get processed track data (smoothed center and size)
            center_x = float(proc_track["x"][fidx])
            center_y = float(proc_track["y"][fidx])
            half_size = float(proc_track["s"][fidx])

            face_data = {
                "track_id": tidx,
                "speaker_id": speaker_id,
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "center": {"x": round(center_x, 2), "y": round(center_y, 2)},
                "size": round(half_size, 2),
                "raw_score": round(float(score[min(fidx, len(score) - 1)]), 3),
                "smoothed_score": round(smoothed_score, 3),
                "is_speaking": is_speaking,
                "speaking_duration": round(
                    speaker_duration_counter.get(speaker_id, 0) / 25.0, 3
                )
                if speaker_id is not None
                else 0.0,
            }

            frames_data[frame_num]["faces"].append(face_data)

    # Build track summary
    tracks_summary = []
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        frame_indices = track["track"]["frame"].tolist()
        speaker_id = speaker_tracks_id_map.get(tidx, None)

        speaking_frames = 0
        for fidx in range(len(frame_indices)):
            s = score[max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)]
            if numpy.mean(s) >= args.speakerThresh:
                speaking_frames += 1

        tracks_summary.append(
            {
                "track_id": tidx,
                "speaker_id": speaker_id,
                "frame_range": [int(frame_indices[0]), int(frame_indices[-1])],
                "total_frames": len(frame_indices),
                "speaking_frames": speaking_frames,
                "speaking_ratio": round(speaking_frames / len(frame_indices), 3),
                "avg_score": round(float(numpy.mean(score)), 3),
            }
        )

    # Build output structure
    output = {
        "video_name": args.videoName,
        "fps": 25,
        "total_frames": total_frames,
        "total_tracks": len(tracks),
        "total_speakers": len(speaker_track_indices),
        "parameters": {
            "speaker_threshold": args.speakerThresh,
            "min_speech_length": args.minSpeechLen,
        },
        "tracks_summary": tracks_summary,
        "frames": frames_data,
    }

    output_path = os.path.join(args.pyworkPath, "frame_metadata.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    sys.stderr.write(
        time.strftime("%Y-%m-%d %H:%M:%S") + f" Frame metadata saved to {output_path}\n"
    )

    return output
