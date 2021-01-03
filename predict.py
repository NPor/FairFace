from __future__ import print_function, division
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import os.path
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import dlib
import os
import argparse
import gdown
import zipfile

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def detect_face(image_path, default_max_size=800,size = 300, padding = 0.25):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('dlib_models/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('dlib_models/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height
    img = dlib.load_rgb_image(image_path)

    old_height, old_width, _ = img.shape

    if old_width > old_height:
        new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
    else:
        new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
    img = dlib.resize_image(img, rows=new_height, cols=new_width)

    dets = cnn_face_detector(img, 1)
    num_faces = len(dets)
    if num_faces == 0:
        raise AttributeError("Sorry, there were no faces found.")
    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in dets:
        rect = detection.rect
        faces.append(sp(img, rect))
    return dlib.get_face_chips(img, faces, size=size, padding = padding)


def ensure_dir(directory):
    if not os.path.exists(directory):

        # fair
        outputs = model_fair_7(image)
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)

        race_outputs = outputs[:7]
        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
        gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
        age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

        race_pred = np.argmax(race_score)
        gender_pred = np.argmax(gender_score)
        age_pred = np.argmax(age_score)

        race_scores_fair.append(race_score)
        gender_scores_fair.append(gender_score)
        age_scores_fair.append(age_score)

        race_preds_fair.append(race_pred)
        os.makedirs(directory)


class AgeGenderRace():
    def __init__(self, cachepath: Path):
        # todo: take the path(line87) an object path
        url = "https://drive.google.com/uc?id=1Wy3uK_7KpkAvaYxNdeZQfSeC5iF9j-x9"
        checkpoint = str((cachepath / "fairface_alldata_20191111.pt").absolute())
        print(1)
        if not os.path.exists(checkpoint):
            gdown.download(url, cachepath / "fair_face_models.zip", quiet=False)
            with zipfile.ZipFile(cachepath / "fair_face_models.zip", 'r') as zip_ref:
                zip_ref.extractall(cachepath)
        print(2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_fair_7 = torchvision.models.resnet34(pretrained=True)
        model_fair_7.fc = nn.Linear(model_fair_7.fc.in_features, 18)
        alldatapath = str(cachepath / "fair_face_models/fairface_alldata_20191111.pt")
        model_fair_7.load_state_dict(torch.load(alldatapath))
        self.model_fair_7 = model_fair_7.to(self.device)
        self.model_fair_7.eval()

        model_fair_4 = torchvision.models.resnet34(pretrained=True)
        model_fair_4.fc = nn.Linear(model_fair_4.fc.in_features, 18)
        alldatapath_race = cachepath / "fair_face_models/fairface_alldata_4race_20191111.pt"
        model_fair_4.load_state_dict(torch.load(alldatapath_race))
        self.model_fair_4 = model_fair_4.to(self.device)
        self.model_fair_4.eval()

        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    def predidct(self, faces):
        # img pth of face images
        face_names = []
        # list within a list. Each sublist contains scores for all races. Take max for predicted race
        race_scores_fair = []
        gender_scores_fair = []
        age_scores_fair = []
        race_preds_fair = []
        gender_preds_fair = []
        age_preds_fair = []
        race_scores_fair_4 = []
        race_preds_fair_4 = []

        for index, image in enumerate(faces):
            image = self.trans(image)
            image = image.view(1, 3, 224, 224)  # reshape image to match model dimensions (1 batch size)
            image = image.to(self.device)

            # fair
            outputs = self.model_fair_7(image)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)

            race_outputs = outputs[:7]
            gender_outputs = outputs[7:9]
            age_outputs = outputs[9:18]

            race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
            gender_score = np.exp(gender_outputs) / np.sum(np.exp(gender_outputs))
            age_score = np.exp(age_outputs) / np.sum(np.exp(age_outputs))

            race_pred = np.argmax(race_score)
            gender_pred = np.argmax(gender_score)
            age_pred = np.argmax(age_score)

            race_scores_fair.append(race_score)
            gender_scores_fair.append(gender_score)
            age_scores_fair.append(age_score)

            race_preds_fair.append(race_pred)
            gender_preds_fair.append(gender_pred)
            age_preds_fair.append(age_pred)

            # fair 4 class
            outputs = self.model_fair_4(image)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.squeeze(outputs)

            race_outputs = outputs[:4]
            race_score = np.exp(race_outputs) / np.sum(np.exp(race_outputs))
            race_pred = np.argmax(race_score)
            race_scores_fair_4.append(race_score)
            race_preds_fair_4.append(race_pred)

        result = pd.DataFrame([face_names,
                               race_preds_fair,
                               race_preds_fair_4,
                               gender_preds_fair,
                               age_preds_fair,
                               race_scores_fair, race_scores_fair_4,
                               gender_scores_fair,
                               age_scores_fair, ]).T
        result.columns = ['face_name_align',
                          'race_preds_fair',
                          'race_preds_fair_4',
                          'gender_preds_fair',
                          'age_preds_fair',
                          'race_scores_fair',
                          'race_scores_fair_4',
                          'gender_scores_fair',
                          'age_scores_fair']
        result.loc[result['race_preds_fair'] == 0, 'race'] = 'White'
        result.loc[result['race_preds_fair'] == 1, 'race'] = 'Black'
        result.loc[result['race_preds_fair'] == 2, 'race'] = 'Latino_Hispanic'
        result.loc[result['race_preds_fair'] == 3, 'race'] = 'East Asian'
        result.loc[result['race_preds_fair'] == 4, 'race'] = 'Southeast Asian'
        result.loc[result['race_preds_fair'] == 5, 'race'] = 'Indian'
        result.loc[result['race_preds_fair'] == 6, 'race'] = 'Middle Eastern'

        # race fair 4

        result.loc[result['race_preds_fair_4'] == 0, 'race4'] = 'White'
        result.loc[result['race_preds_fair_4'] == 1, 'race4'] = 'Black'
        result.loc[result['race_preds_fair_4'] == 2, 'race4'] = 'Asian'
        result.loc[result['race_preds_fair_4'] == 3, 'race4'] = 'Indian'

        # gender
        result.loc[result['gender_preds_fair'] == 0, 'gender'] = 'Male'
        result.loc[result['gender_preds_fair'] == 1, 'gender'] = 'Female'

        # age
        result.loc[result['age_preds_fair'] == 0, 'age'] = '0-2'
        result.loc[result['age_preds_fair'] == 1, 'age'] = '3-9'
        result.loc[result['age_preds_fair'] == 2, 'age'] = '10-19'
        result.loc[result['age_preds_fair'] == 3, 'age'] = '20-29'
        result.loc[result['age_preds_fair'] == 4, 'age'] = '30-39'
        result.loc[result['age_preds_fair'] == 5, 'age'] = '40-49'
        result.loc[result['age_preds_fair'] == 6, 'age'] = '50-59'
        result.loc[result['age_preds_fair'] == 7, 'age'] = '60-69'
        result.loc[result['age_preds_fair'] == 8, 'age'] = '70+'

        return result[['face_name_align',
                'race', 'race4',
                'gender', 'age',
                'race_scores_fair', 'race_scores_fair_4',
                'gender_scores_fair', 'age_scores_fair']]



if __name__ == "__main__":

    faces = detect_face("./detected_faces/race_Asian_face0.jpg")
    cachepath = Path("./models_cache")

    pagdr = AgeGenderRace(cachepath)

    #Please change test_outputs.csv to actual name of output csv.
    results = pagdr.predidct(faces)

    print(results["race4"])
