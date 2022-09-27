"""
    Defining the different object detection models 
"""
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
import configparser
dtype = torch.float32

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
        

    def CONV(self):
        """
            Defining the convolutional part of YOLOv1
        """
        conv_configs = self.configurations()[1]
        yolo_conv = nn.Sequential()

        for i in range(len(conv_configs)):
            layer = list(conv_configs.keys())[i]

            if layer[:4] == "conv":
                    
                out_filters = int(conv_configs[layer]["filters"])
                kernel = int(conv_configs[layer]["size"])
                stride = int(conv_configs[layer]["stride"])
                pad = int(conv_configs[layer]["pad"])

                if i == 0:              
                    conv_block = nn.Sequential(
                        nn.Conv2d(in_channels=3, out_channels=out_filters, kernel_size=self.S, stride=stride, padding=pad, bias=False, dtype=dtype), 
                        nn.BatchNorm2d(num_features=out_filters),
                        nn.LeakyReLU(),
                    )
                    yolo_conv.append(conv_block)    
                    next_in_filters = out_filters

                else: 
                    conv_block = nn.Sequential(
                        nn.Conv2d(in_channels=next_in_filters, out_channels=out_filters, kernel_size=kernel, stride=stride, padding=pad, dtype=dtype), 
                        nn.BatchNorm2d(num_features=out_filters),
                        nn.LeakyReLU(),
                    )
                    yolo_conv.append(conv_block)
                    next_in_filters = out_filters

            if layer[:4] == "maxp":
                size = int(conv_configs[layer]["size"])
                stride = int(conv_configs[layer]["stride"])

                yolo_conv.append(nn.MaxPool2d(kernel_size=size, stride=stride))

        return yolo_conv


    def CONNECTED(self):
        """
            Defining the connected part of YOLOv1 
        """
        FC = nn.Sequential(
            nn.Linear(in_features=1024*self.S*self.S, out_features=4096, dtype=torch.float32),
            nn.Dropout(0.0),
            nn.LeakyReLU(),
            nn.Linear(in_features=4096, out_features = self.S*self.S*(self.C + 5*self.B))
        )
        return FC 

    def network(self):
        yolo = nn.Sequential(
            self.CONV(),
            nn.Flatten(start_dim=1, end_dim=-1),
            self.CONNECTED()
        )
        return yolo 



if __name__ == "__main__":
    testdata = torch.randn(size=(1, 3, 448, 448), dtype=torch.float32)
    PATH = "/home/agastya123/PycharmProjects/ComputerVision/PennFudan/yolov1.cfg"

    YOLO = YOLOv1(config_path = PATH, in_channels=3, split_size=7, num_boxes=2, num_classes=20)

    CONV = YOLO.CONV()
    FC = YOLO.CONNECTED()
    net = YOLO.network()    

    print(CONV[:2])




    
    





    

    



    


    
    



    