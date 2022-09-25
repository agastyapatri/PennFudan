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
    def __init__(self, img_PATH, annot_PATH):
        super().__init__()
        self.imgpath = img_PATH
        self.annpath = annot_PATH

    def __getitem__(self, idx):

        """
        Returns the image and annotations at the desired index
        """

        # lists of all images and all annotations 
        imgs = list(sorted(os.listdir(self.imgpath)))
        annots = list(sorted(os.listdir(self.annpath)))

        # image is a np array of size HxWxC
        image = cv2.imread(os.path.join(self.imgpath, imgs[idx]))
        
        return annots[idx]





        

    def loadoneimage(self):
        pass 
        


    def __len__(self):
        pass 

    def data(self):
        pass 




if __name__ == "__main__":
    dataset = Database(
        img_PATH="/home/agastya123/PycharmProjects/ComputerVision/datasets/PennFudanPed/PNGImages/", 
        annot_PATH="/home/agastya123/PycharmProjects/ComputerVision/datasets/PennFudanPed/Annotation/"
        )

    print(dataset[0])
    

    