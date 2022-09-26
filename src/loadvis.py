"""
    Defining a dataset class that will wrap around the images and annotations 

    The reference scripts for training object detection, instance segmentation and keypoint detectoin allows for easily supporting adding new custom datasets. 
"""

import torch
import torch.nn as nn  
import matplotlib.pyplot as plt 
import numpy as np 
from PIL import Image 
import cv2 
import os
torch.manual_seed(0)


class Database(torch.utils.data.Dataset):
    """
    :returns tuple: image tensor, dimension tuple, BB_Coordinates 
    """
    def __init__(self, img_PATH, annot_PATH):
        super().__init__()
        self.imgpath = img_PATH
        self.annpath = annot_PATH
        
        self.imgs = list(sorted(os.listdir(self.imgpath)))
        self.annots = list(sorted(os.listdir(self.annpath)))

        self.classes = {
            0 : "Background",
            1 : "PASpersonWalking"
        }

    def __getitem__(self, idx):

        """
        Returns the image and annotations at the desired index
        """
        ann = self.annots[idx]        
        img = cv2.imread(os.path.join(self.imgpath, self.imgs[idx]))

        # Images: H x W x C
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        width, height = image.shape[0], image.shape[1]        

        # Annotations
        file = open(os.path.join(self.annpath, ann)) 
        lines = file.readlines()
        file.close()
        imp_lines = [] 
        annotations = [] 

        for line in lines:
            if line[0] != "#" and line != "\n":
                prop, item = line.split(":")
                if prop[0] == "B":
                    item = item[:-1].split(" - ")

                    ann = {
                        "min" : item[0].split(" ", maxsplit=1)[1],
                        "max" : item[1]
                        }

                    annotations.append((prop, ann))            
        
        annotations = dict(annotations)        
        return image, (height, width), annotations  



if __name__ == "__main__":
    dataset = Database(
        img_PATH="/home/agastya123/PycharmProjects/ComputerVision/datasets/PennFudanPed/PNGImages/", 
        annot_PATH="/home/agastya123/PycharmProjects/ComputerVision/datasets/PennFudanPed/Annotation/"
        )

    print(dataset[0])
