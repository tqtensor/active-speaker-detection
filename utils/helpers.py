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
