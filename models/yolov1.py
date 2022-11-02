"""
    Defining the different object detection models 
"""
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
import configparser
dtype = torch.float32
import json 
from collections import OrderedDict


class CONV(nn.Module):
    #   Class Defining a singular convolutional block 
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dtype=dtype)
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, input):
        block = nn.Sequential(
            self.conv, 
            self.batchnorm,
            self.activation
        )

        return block(input)





class YOLOv1(nn.Module):
    """
    Class to define the YOLO object detection system
    """
    def __init__(self, config_path, in_channels, split_size, num_boxes, num_classes):

        super().__init__()
        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.PATH = config_path
        
        # loading the configurations of YOLOv1 
        self.config = configparser.ConfigParser()
        self.config.read(self.PATH)



    def __str__(self):

        params = f"\nYOLOv1 Object Detection.\n-------------------------\nSplit Size: {self.S}\nNumber of Boxes: {self.B}\nNumber of Classes: {self.C}\n"
        return params 
    

    def configurations(self):
        """
            Redmon et al. configurations for the network, convolutions, classification, detection.
        """
        conv_config = {} 
        net_config = {}
        conn_config = {}
        

        for sect in self.config.sections():
            
            #populating the configurations for the convolutional blocks 
            if sect[:4] == "conv":
                conv_config[sect] =  {
                    "batch_normalize" : self.config.get(section=sect, option="batch_normalize"),
                    "filters" : self.config.get(section=sect, option="filters"),
                    "size" : self.config.get(section=sect, option="size"),
                    "stride": self.config.get(section=sect, option="stride"),
                    "pad": self.config.get(section=sect, option="pad"),
                    "activation": self.config.get(section=sect, option="activation")
                    }

            # populating the configurations for the maxpool layers
            elif sect[:4] == "maxp":
                conv_config[sect] =  {
                    "size" : self.config.get(section=sect, option="size"), 
                    "stride": self.config.get(section=sect, option="stride"),
                    }


            # Network Hyperparameter Configurations
            elif sect[:3] == "net":
                for option in self.config.options(sect):
                    net_config[option] = self.config.get(section=sect, option=option)

            elif sect[:4] == "conn" or sect[:4] == "dete":
                for option in self.config.options(sect):
                    conn_config[option] = self.config.get(section=sect, option=option)
                    
        return net_config, conv_config, conn_config
        

    def CONVOLUTIONAL(self):
        """
            Defining the convolutional part of YOLOv1
        """
        conv_configs = self.configurations()[1]
        input_channels = 3
        darknet = nn.Sequential()
        in_channels = 3
        for i in conv_configs:

            if i[:4] == "conv":
                # Convolutional layer configs
                out_channels = int(conv_configs[i]["filters"])
                size = int(conv_configs[i]["size"])
                stride = int(conv_configs[i]["stride"])
                pad = int(conv_configs[i]["pad"])

                conv_block = CONV(in_channels=in_channels, out_channels=out_channels, kernel_size=size, stride=stride, padding = pad)
                in_channels = out_channels
                darknet.append(conv_block)

            else:
                # Maxpool layer configs
                darknet.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return darknet



    def CONNECTED(self):
        """
            Defining the connected part of YOLOv1 
        """
        FC = nn.Sequential(
            nn.Linear(in_features=1024*self.S*self.S, out_features=4096, dtype=torch.float32),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=4096, out_features = self.S*self.S*(self.C + 5*self.B))
        )
        
        return FC 

    def network(self):
        yolo = nn.Sequential(
            self.CONVOLUTIONAL(),
            nn.Flatten(),
            self.CONNECTED()
        )
        return yolo 



if __name__ == "__main__":
    testdata = torch.randn(size=(1, 3, 448, 448), dtype=torch.float32)
    PATH = "/home/agastyapatri/Projects/ComputerVision/PennFudan/yolov1.cfg"

    YOLO = YOLOv1(config_path = PATH, in_channels=3, split_size=7, num_boxes=2, num_classes=20)

    FC = YOLO.CONNECTED()
    CONVLUTIONAL = YOLO.CONVOLUTIONAL()    
    
    darknet = YOLO.network()

    





    
    





    

    



    


    
    



    