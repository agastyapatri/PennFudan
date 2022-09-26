import torch 
import cv2 
import numpy as np 
from torchvision.models import detection 
import pickle
from src.loadvis import Database
from src.yolov1 import YOLOv1



path_to_images = "/home/agastya123/PycharmProjects/ComputerVision/datasets/PennFudanPed/PNGImages/"
path_to_annotations = "/home/agastya123/PycharmProjects/ComputerVision/datasets/PennFudanPed/Annotation/" 
path_to_yolo_configs = "/home/agastya123/PycharmProjects/ComputerVision/ChessPieces/yolov1.cfg"

# Dataset is a tuple of images, target BB coordinates and image dimensions   
dataset = Database(
        img_PATH = path_to_images,
        annot_PATH = path_to_annotations
        )



