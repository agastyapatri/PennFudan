"""
    Defining the different object detection models 
"""
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)


model = torch.hub.load("ultralytics/yolov5", "yolov5s")
