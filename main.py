import os
import glob
import pickle
import warnings
import subprocess
from shutil import rmtree
import tqdm

# Local imports
from config.args import get_args
from model.yoloFace import run_face_detection
from utils.video_utils import extract_video, extract_audio, extract_frames
from utils.track_utils import scene_detect, track_shot, crop_video
from utils.inference_utils import get_speaker_track_indices, evaluate_network
from utils.helpers import visualization, summarize_tracks

warnings.filterwarnings("ignore")


def download_weights(args):
    if not os.path.isfile(args.talkNetWeights):
        os.makedirs(os.path.dirname(args.talkNetWeights), exist_ok=True)
        subprocess.run(["gdown", "--id", "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea", "-O", args.talkNetWeights], stdout=subprocess.DEVNULL)

    if not os.path.isfile(args.yoloFaceWeights):
        os.makedirs(os.path.dirname(args.yoloFaceWeights), exist_ok=True)
        subprocess.run([
            "gdown", "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov11n-face.pt",
            "-O", args.yoloFaceWeights
        ], stdout=subprocess.DEVNULL)


def prepare_paths(args):
    args.videoPath = glob.glob(os.path.join(args.videoFolder, args.videoName + '.*'))[0]
    args.savePath = os.path.join(args.videoFolder, args.videoName)
    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')

    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.pyaviPath)
    os.makedirs(args.pyframesPath)
    os.makedirs(args.pyworkPath)
    os.makedirs(args.pycropPath)


def main():
    args = get_args()
    download_weights(args) # Download weights if not present
    prepare_paths(args)

    # Step 1: Preprocess
    extract_video(args)
    extract_audio(args)
    extract_frames(args)

    # Step 2: Face detection
    scene = scene_detect(args)
    faces = run_face_detection(args)

    # Step 3: Face tracking
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num]))

    for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d' % ii)))

    with open(os.path.join(args.pyworkPath, 'tracks.pckl'), 'wb') as f:
        pickle.dump(vidTracks, f)

    # Step 4: ASD inference
    files = sorted(glob.glob(f"{args.pycropPath}/*.avi"))
    scores = evaluate_network(files, args)
    with open(os.path.join(args.pyworkPath, 'scores.pckl'), 'wb') as f:
        pickle.dump(scores, f)

    # Step 5: Speaker processing
    speaker_track_indices = get_speaker_track_indices(scores, args)
    visualization(vidTracks, scores, args, speaker_track_indices)
    summarize_tracks(vidTracks, scores, args, speaker_track_indices)


if __name__ == '__main__':
    main()
