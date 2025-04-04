import cv2
import os
import glob
import pickle
import sys
from ultralytics import YOLO

def load_yolo_model(weights_path):
    return YOLO(weights_path)

def run_face_detection(model, frames_path, video_file_path, work_path):
    flist = sorted(glob.glob(os.path.join(frames_path, '*.jpg')))
    dets = []

    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        results = model.predict(image, conf=0.7, iou=0.5)
        dets.append([
            {
                'frame': fidx,
                'bbox': box.xyxy.cpu().numpy().tolist()[0],
                'conf': float(box.conf.item())
            }
            for box in results[0].boxes
        ])
        sys.stderr.write('%s-%05d; %d dets\r' % (video_file_path, fidx, len(dets[-1])))

    with open(os.path.join(work_path, 'faces.pckl'), 'wb') as fil:
        pickle.dump(dets, fil)

    return dets

def run_face_detection(args):
    model = YOLO(args.yoloFaceWeights)

    flist = sorted(glob.glob(os.path.join(args.pyframesPath, '*.jpg')))
    dets = []

    for fidx, fname in enumerate(flist):
        image = cv2.imread(fname)
        results = model.predict(image, conf=0.7, iou=0.5)
        dets.append([
            {'frame': fidx, 'bbox': box.xyxy.cpu().numpy().tolist()[0], 'conf': float(box.conf.item())}
            for box in results[0].boxes
        ])
        sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))

    with open(os.path.join(args.pyworkPath, 'faces.pckl'), 'wb') as fil:
        pickle.dump(dets, fil)

    return dets
