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
    dataset[idx1][0]: returns the image tensor of the idx1 image
    dataset[idx][1]: returns the (H x W) tuple of the idx1 image
    dataset[idx1][2]: returns the annotations of the idx1 image
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
        Returns the 
        """
        ann = self.annots[idx]        
        img = cv2.imread(os.path.join(self.imgpath, self.imgs[idx]))

        # Images: W x H x C
        image = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32))
        width, height = image.shape[0], image.shape[1]        

        # Annotations
        file = open(os.path.join(self.annpath, ann)) 
        lines = file.readlines()
        file.close()

        targets = [] 
        
        # Creating the target dictionary. targets is (x_min, y_min), (x_max, y_max) 
        j = 0
        
        # convert_to_tuple = lambda s: list(s)[]
        
        for line in lines:
            if line[0] != "#" and line != "\n":
                prop, item = line.split(":")
                if prop[0] == "B":
                    item = item[:-1].split(" - ")
                    min_coords = item[0].split(" ", maxsplit=1)[1]
                    max_coords = item[1]

                    ann = {
                        "x_min" : int("".join(list(min_coords)[1:4])),
                        "y_min" : int("".join(list(min_coords)[6:-1])),
                        "x_max" : int("".join(list(max_coords)[1:4])),
                        "y_max" : int("".join(list(min_coords)[6:-1])),
                    }
                    prop = f"BB_{j}"
                    j+=1
                    targets.append((prop, ann))            
        
        targets = dict(targets)        
        return image, (height, width), targets  


    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    dataset = Database(
        img_PATH="/home/agastya123/PycharmProjects/ComputerVision/datasets/PennFudanPed/PNGImages/", 
        annot_PATH="/home/agastya123/PycharmProjects/ComputerVision/datasets/PennFudanPed/Annotation/"
        )

    # print(dataset[5][2]["Bounding box for object 2 \"PASpersonWalking\" (Xmin, Ymin) - (Xmax, Ymax) "]["min"])
