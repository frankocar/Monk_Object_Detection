import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from src.model import EfficientDet
from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.autonotebook import tqdm
from src.config import colors
import cv2

from sklearn.preprocessing import minmax_scale


class Infer():
    '''
    Class for main inference

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {}
        self.system_dict["verbose"] = verbose
        self.system_dict["local"] = {}
        self.system_dict["local"]["common_size"] = 1024
        self.system_dict["local"]["mean"] = np.array([[[0.5, 0.5, 0.5]]])
        self.system_dict["local"]["std"] = np.array([[[0.3, 0.3, 0.3]]])
        # self.system_dict["local"]["mean"] = np.array([[[0.485, 0.456, 0.406]]])
        # self.system_dict["local"]["std"] = np.array([[[0.229, 0.224, 0.225]]])

    def Model(self, model_dir="trained/"):
        '''
        User function: Selet trained model params

        Args:
            model_dir (str): Relative path to directory containing trained models 

        Returns:
            None
        '''
        self.system_dict["local"]["model"] = torch.load(model_dir + "/signatrix_efficientdet_coco.pth", map_location=torch.device('cpu')).module
        if torch.cuda.is_available():
            self.system_dict["local"]["model"] = self.system_dict["local"]["model"].cuda();

    def Predict(self, img_path):
        '''
        User function: Run inference on image and visualize it

        Args:
            img_path (str): Relative path to the image file

        Returns:
            tuple: Contaning label IDs, Scores and bounding box locations of predicted objects. 
        '''

        with np.load(img_path) as data:
            if len(data) != 1 or "arr_0" not in data:
                raise Exception("More than 1 array in the npz or name invalid")
            arr = data['arr_0'].astype(np.float32)

        if arr.shape[0] != arr.shape[1]:
            print("ERROR", img_path)
               
        data = minmax_scale(np.clip(arr, 0, 99999), feature_range=(0, 1))
        image = np.repeat(data[:, :, np.newaxis], 3, axis=2)

        image = (image.astype(np.float32) - self.system_dict["local"]["mean"]) / self.system_dict["local"]["std"]
        height, width, _ = image.shape

        if height != self.system_dict["local"]["common_size"]:
            if height > width:
                scale = self.system_dict["local"]["common_size"] / height
                resized_height = self.system_dict["local"]["common_size"]
                resized_width = int(width * scale)
            else:
                scale = self.system_dict["local"]["common_size"] / width
                resized_height = int(height * scale)
                resized_width = self.system_dict["local"]["common_size"]

            image = cv2.resize(image, (resized_width, resized_height))

            new_image = np.zeros((self.system_dict["local"]["common_size"], self.system_dict["local"]["common_size"], 3))
            new_image[0:resized_height, 0:resized_width] = image
        else:
            new_image = np.zeros((self.system_dict["local"]["common_size"], self.system_dict["local"]["common_size"], 3))
            new_image[0:self.system_dict["local"]["common_size"], 0:self.system_dict["local"]["common_size"]] = image

        img = torch.from_numpy(new_image)

        with torch.no_grad():
            scores, labels, boxes = self.system_dict["local"]["model"](img.cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
            if height != self.system_dict["local"]["common_size"]:
                boxes /= scale;

        return scores, labels, boxes

    def Predict_direct(self, img_data):
        if img_data.shape[0] != img_data.shape[1]:
            print("ERROR")
               
        data = minmax_scale(np.clip(img_data, 0, 99999), feature_range=(0, 1))
        image = np.repeat(data[:, :, np.newaxis], 3, axis=2)

        image = (image.astype(np.float32) - self.system_dict["local"]["mean"]) / self.system_dict["local"]["std"]
        height, width, _ = image.shape

        if height != self.system_dict["local"]["common_size"]:
            if height > width:
                scale = self.system_dict["local"]["common_size"] / height
                resized_height = self.system_dict["local"]["common_size"]
                resized_width = int(width * scale)
            else:
                scale = self.system_dict["local"]["common_size"] / width
                resized_height = int(height * scale)
                resized_width = self.system_dict["local"]["common_size"]

            image = cv2.resize(image, (resized_width, resized_height))

            new_image = np.zeros((self.system_dict["local"]["common_size"], self.system_dict["local"]["common_size"], 3))
            new_image[0:resized_height, 0:resized_width] = image
        else:
            new_image = np.zeros((self.system_dict["local"]["common_size"], self.system_dict["local"]["common_size"], 3))
            new_image[0:self.system_dict["local"]["common_size"], 0:self.system_dict["local"]["common_size"]] = image

        img = torch.from_numpy(new_image)

        with torch.no_grad():
            scores, labels, boxes = self.system_dict["local"]["model"](img.cuda().permute(2, 0, 1).float().unsqueeze(dim=0))
            if height != self.system_dict["local"]["common_size"]:
                boxes /= scale

        return scores, labels, boxes
        