"""
    Defining the different object detection models 
"""
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)

class YOLOv1(nn.Module):
    """You Only Look Once, Ver 1"""
    
    def __init__(self, input_channels, split_size, num_classes, num_boxes):
        super().__init__()
        self.input_channels = input_channels
        self.S = split_size
        self.C = num_classes
        self.B = num_boxes

    def CONV(self):
        # Defining the convolutional part of the model
        Block1 = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=self.S, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        Block2 = nn.Sequential(
            nn.Conv2d(64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        Block3 = nn.Sequential(
            nn.Conv2d(192, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        Block4 = nn.Sequential(
            nn.Conv2d(512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2)
        )

        Block5 = nn.Sequential(
            nn.Conv2d(1024, out_channels=512, kernel_size=1, stride=1, padding=1),
            nn.Conv2d(512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, out_channels=512, kernel_size=1, stride=1, padding=1),
            nn.Conv2d(512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        )
        Block6 = nn.Sequential(
            nn.Conv2d(1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, out_channels=1024, kernel_size=3, stride=1, padding=1)
        )

        conv = nn.Sequential(
            Block1,
            Block2,
            Block3,
            Block4,
            Block5,
            Block6
            )
        return conv 


    def FC(self):
        fc = nn.Sequential(
            nn.Linear(in_features=1024*self.S*self.S, out_features=496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=496, out_features=self.S*self.S*(self.C + self.B*5))
        )
        return fc 



    def network(self):
        CONV = self.CONV()
        FC = self.FC() 

        net = nn.Sequential(
            CONV, 
            nn.Flatten(),
            FC
        )
        return net
    






if __name__ == "__main__":
    yolo = YOLOv1(input_channels=3, split_size = 7, num_classes=20, num_boxes=2)

    # data in CxHxW.
    example_Data = torch.randn(3, 448, 448)
    darknet = yolo.network()
    example_output = darknet(example_Data)
    print(example_output.shape)
    


    
    



    