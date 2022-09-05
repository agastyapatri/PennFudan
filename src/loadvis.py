"""
    Class to load the relevant data 
"""
import torch 
import torch.nn as nn 
import numpy as np 
torch.manual_seed(0)
from torch.utils.data import DataLoader 
from torchvision import datasets, transforms


class Loader(nn.Module):
    
    def __init__(self, PATH, batch_size):
        super().__init__()
        self.path = PATH 
        self.batch_size = batch_size

    def getPASCAL(self):
        """
        Getting the MS COCO detection dataset
        """
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0,0,0), std=(1,1,1))
        ])
        trainset = datasets.VOCDetection(root=self.path, year='2007', image_set="train", download=False, transform=trans)

        testset = datasets.VOCDetection(root=self.path, year='2007', image_set="test", download=False, transform=trans)

        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=True)
        
        return len(trainloader), len(testloader)  



if __name__ == "__main__":
    
    loadcoco = Loader(PATH="/home/agastya123/PycharmProjects/ComputerVision/datasets/PASCAL-VOC/2006/", batch_size=1)

    print(loadcoco.getPASCAL())

