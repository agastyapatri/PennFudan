import torch 
import numpy as np 
from torchvision.models import detection 
from src.loadvis import Database
from models.yolov1 import YOLOv1
from src.utils import Utilities



# Defining path variables for configs and data. 
path_to_images = "/home/agastyapatri/Projects/ComputerVision/datasets/PennFudanPed/PNGImages/"
path_to_annotations = "/home/agastyapatri/Projects/ComputerVision/datasets/PennFudanPed/Annotation/" 
path_to_yolo_configs = "/home/agastyapatri/Projects/ComputerVision/PennFudan/yolov1.cfg"

testdata = torch.randn(size=(1, 3, 448, 448), dtype=torch.float32)

# Dataset is a tuple of images, image dimensions and target BB coordinates    
dataset = Database(
        img_PATH = path_to_images,
        annot_PATH = path_to_annotations
        )

# model used is YOLOv1 as defined by Redmon et al. 2 classes, as PennFudan has background = 0, and pedestrian = 1. 
YOLO = YOLOv1(config_path=path_to_yolo_configs, in_channels=3, split_size=7, num_boxes=2, num_classes=2)

network = YOLO.network()

print(network)