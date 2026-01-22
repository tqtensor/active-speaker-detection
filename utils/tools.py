import glob
import os
import subprocess

import cv2
import numpy
import pandas
import tqdm
from scipy.io import wavfile


def init_args(args):
    """Initializes argument paths for AVA dataset processing.

    Sets up various directory paths for model saving, scoring, trial data,
    audio/video paths, and evaluation configurations based on the provided
    arguments. The details for these folders/files are described in the
    preprocess_AVA function below.

    Args:
        args: Configuration object containing savePath, dataPathAVA, and
            evalDataType attributes.

    Returns:
        Modified args object with initialized path attributes.
    """
    args.modelSavePath = os.path.join(args.savePath, "model")
    args.scoreSavePath = os.path.join(args.savePath, "score.txt")
    args.trialPathAVA = os.path.join(args.dataPathAVA, "csv")
    args.audioOrigPathAVA = os.path.join(args.dataPathAVA, "orig_audios")
    args.visualOrigPathAVA = os.path.join(args.dataPathAVA, "orig_videos")
    args.audioPathAVA = os.path.join(args.dataPathAVA, "clips_audios")
    args.visualPathAVA = os.path.join(args.dataPathAVA, "clips_videos")
    args.trainTrialAVA = os.path.join(args.trialPathAVA, "train_loader.csv")

    if args.evalDataType == "val":
        args.evalTrialAVA = os.path.join(args.trialPathAVA, "val_loader.csv")
        args.evalOrig = os.path.join(args.trialPathAVA, "val_orig.csv")
        args.evalCsvSave = os.path.join(args.savePath, "val_res.csv")
    else:
        args.evalTrialAVA = os.path.join(args.trialPathAVA, "test_loader.csv")
        args.evalOrig = os.path.join(args.trialPathAVA, "test_orig.csv")
        args.evalCsvSave = os.path.join(args.savePath, "test_res.csv")

    os.makedirs(args.modelSavePath, exist_ok=True)
    os.makedirs(args.dataPathAVA, exist_ok=True)
    return args


def download_pretrain_model_AVA():
    """Downloads the pretrained AVA model if not already present.

    Downloads the pretrained model file from Google Drive using gdown
    if the file 'pretrain_AVA.model' does not exist in the current directory.
    """
    if not os.path.isfile("pretrain_AVA.model"):
        link = "1KwTIA0XOl71DswvYCYGMLHSrxufwkEh8"
        cmd = "gdown --id %s -O %s" % (link, "pretrain_AVA.model")
        subprocess.call(cmd, shell=True, stdout=None)


def preprocess_AVA(args):
    """Preprocesses the AVA dataset for active speaker detection.

    This preprocessing is modified based on the repository at
    https://github.com/fuankarion/active-speakers-context.

    The process requires 302 GB of space initially. If space is limited:
    - Delete orig_videos (167GB) after generating clips_videos (85GB)
    - Delete orig_audios (44GB) after generating clips_audios (6.4GB)
    Final space requirement is less than 100GB.

    The AVA dataset will be saved in the dataPathAVA folder with the
    following structure:
    - clips_audios/: Audio clips cut from original movies
      - test/, train/, val/
    - clips_videos/: Face clips saved as images frame-by-frame
      - test/, train/, val/
    - csv/: CSV files for data loading
      - test_file_list.txt, test_loader.csv, test_orig.csv
      - train_loader.csv, train_orig.csv
      - trainval_file_list.txt
      - val_loader.csv, val_orig.csv
    - orig_audios/: Original audios from movies
      - test/, trainval/
    - orig_videos/: Original movies
      - test/, trainval/

    Args:
        args: Configuration object with dataPathAVA attribute.
    """
    download_csv(args)  # Take 1 minute
    download_videos(args)  # Take 6 hours
    extract_audio(args)  # Take 1 hour
    extract_audio_clips(args)  # Take 3 minutes
    extract_video_clips(args)  # Take about 2 days


def download_csv(args):
    """Downloads and extracts required CSV files for AVA dataset.

    Takes approximately 1 minute to download the required CSV files from
    Google Drive and extract them to the data directory.

    Args:
        args: Configuration object with dataPathAVA attribute.
    """
    link = "1CY5ASSAnv0E57uQtd2dfhZRHn2OxzFK8"
    cmd = "gdown --id %s -O %s" % (link, args.dataPathAVA + "/csv.tar.gz")
    subprocess.call(cmd, shell=True, stdout=None)
    cmd = "tar -xzvf %s -C %s" % (args.dataPathAVA + "/csv.tar.gz", args.dataPathAVA)
    subprocess.call(cmd, shell=True, stdout=None)
    os.remove(args.dataPathAVA + "/csv.tar.gz")


def download_videos(args):
    """Downloads original AVA movie files from AWS.

    Takes approximately 6 hours to download the original movies from the
    AVA dataset repository hosted on AWS S3. Downloads both trainval and
    test datasets following the repository at
    https://github.com/cvdfoundation/ava-dataset.

    Args:
        args: Configuration object with trialPathAVA and visualOrigPathAVA
            attributes.
    """
    for dataType in ["trainval", "test"]:
        fileList = (
            open("%s/%s_file_list.txt" % (args.trialPathAVA, dataType))
            .read()
            .splitlines()
        )
        outFolder = "%s/%s" % (args.visualOrigPathAVA, dataType)
        for fileName in fileList:
            cmd = "wget -P %s https://s3.amazonaws.com/ava-dataset/%s/%s" % (
                outFolder,
                dataType,
                fileName,
            )
            subprocess.call(cmd, shell=True, stdout=None)


def extract_audio(args):
    """Extracts audio tracks from video files.

    Takes approximately 1 hour to extract audio from all movies in the
    dataset. Converts audio to 16kHz mono WAV format using PCM signed
    16-bit little-endian encoding.

    Args:
        args: Configuration object with visualOrigPathAVA and
            audioOrigPathAVA attributes.
    """
    for dataType in ["trainval", "test"]:
        inpFolder = "%s/%s" % (args.visualOrigPathAVA, dataType)
        outFolder = "%s/%s" % (args.audioOrigPathAVA, dataType)
        os.makedirs(outFolder, exist_ok=True)
        videos = glob.glob("%s/*" % (inpFolder))
        for videoPath in tqdm.tqdm(videos):
            audioPath = "%s/%s" % (
                outFolder,
                videoPath.split("/")[-1].split(".")[0] + ".wav",
            )
            cmd = (
                "ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 %s -loglevel panic"
                % (videoPath, audioPath)
            )
            subprocess.call(cmd, shell=True, stdout=None)


def extract_audio_clips(args):
    """Extracts audio clips for each entity from the full audio files.

    Takes approximately 3 minutes to extract audio clips corresponding to
    each entity's time segment from the original audio files. Processes
    train, validation, and test datasets, organizing clips by video ID
    and entity ID.

    Args:
        args: Configuration object with trialPathAVA, audioPathAVA, and
            audioOrigPathAVA attributes.
    """
    dic = {"train": "trainval", "val": "trainval", "test": "test"}
    for dataType in ["train", "val", "test"]:
        df = pandas.read_csv(
            os.path.join(args.trialPathAVA, "%s_orig.csv" % (dataType)), engine="python"
        )
        dfNeg = pandas.concat([df[df["label_id"] == 0], df[df["label_id"] == 2]])
        dfPos = df[df["label_id"] == 1]
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(["entity_id", "frame_timestamp"]).reset_index(drop=True)
        entityList = df["entity_id"].unique().tolist()
        df = df.groupby("entity_id")
        audioFeatures = {}
        outDir = os.path.join(args.audioPathAVA, dataType)
        audioDir = os.path.join(args.audioOrigPathAVA, dic[dataType])
        for video_id in df["video_id"].unique().tolist():
            d = os.path.join(outDir, video_id[0])
            if not os.path.isdir(d):
                os.makedirs(d)
        for entity in tqdm.tqdm(entityList, total=len(entityList)):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]["video_id"]
            start = insData.iloc[0]["frame_timestamp"]
            end = insData.iloc[-1]["frame_timestamp"]
            entityID = insData.iloc[0]["entity_id"]
            insPath = os.path.join(outDir, videoKey, entityID + ".wav")
            if videoKey not in audioFeatures.keys():
                audioFile = os.path.join(audioDir, videoKey + ".wav")
                sr, audio = wavfile.read(audioFile)
                audioFeatures[videoKey] = audio
            audioStart = int(float(start) * sr)
            audioEnd = int(float(end) * sr)
            audioData = audioFeatures[videoKey][audioStart:audioEnd]
            wavfile.write(insPath, sr, audioData)


def extract_video_clips(args):
    """Extracts face clips from video files for each entity.

    Takes approximately 2 days to crop face clips from all videos. If you
    only need train and validation data, processing time is reduced to 1 day.
    This process may generate many warning messages which can be safely ignored.

    For each entity, extracts face regions frame-by-frame based on bounding
    box coordinates and saves them as JPG images organized by video ID and
    entity ID.

    Args:
        args: Configuration object with trialPathAVA, visualPathAVA, and
            visualOrigPathAVA attributes.
    """
    # You can optimize this code to save time, while this process is one-time.
    # If you do not need the data for the test set, you can only deal with the train and val part. That will take 1 day.
    # This procession may have many warning info, you can just ignore it.
    dic = {"train": "trainval", "val": "trainval", "test": "test"}
    for dataType in ["train", "val", "test"]:
        df = pandas.read_csv(
            os.path.join(args.trialPathAVA, "%s_orig.csv" % (dataType))
        )
        dfNeg = pandas.concat([df[df["label_id"] == 0], df[df["label_id"] == 2]])
        dfPos = df[df["label_id"] == 1]
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(["entity_id", "frame_timestamp"]).reset_index(drop=True)
        entityList = df["entity_id"].unique().tolist()
        df = df.groupby("entity_id")
        outDir = os.path.join(args.visualPathAVA, dataType)
        for video_id in df["video_id"].unique().tolist():
            d = os.path.join(outDir, video_id[0])
            if not os.path.isdir(d):
                os.makedirs(d)
        for entity in tqdm.tqdm(entityList, total=len(entityList)):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]["video_id"]
            entityID = insData.iloc[0]["entity_id"]
            videoDir = os.path.join(args.visualOrigPathAVA, dic[dataType])
            videoFile = glob.glob(os.path.join(videoDir, "{}.*".format(videoKey)))[0]
            V = cv2.VideoCapture(videoFile)
            insDir = os.path.join(os.path.join(outDir, videoKey, entityID))
            if not os.path.isdir(insDir):
                os.makedirs(insDir)
            j = 0
            for _, row in insData.iterrows():
                imageFilename = os.path.join(
                    insDir, str("%.2f" % row["frame_timestamp"]) + ".jpg"
                )
                V.set(cv2.CAP_PROP_POS_MSEC, row["frame_timestamp"] * 1e3)
                _, frame = V.read()
                h = numpy.size(frame, 0)
                w = numpy.size(frame, 1)
                x1 = int(row["entity_box_x1"] * w)
                y1 = int(row["entity_box_y1"] * h)
                x2 = int(row["entity_box_x2"] * w)
                y2 = int(row["entity_box_y2"] * h)
                face = frame[y1:y2, x1:x2, :]
                j = j + 1
                cv2.imwrite(imageFilename, face)
