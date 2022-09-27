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
        Method to load and sort the configuratins in self.config
        :return net_config: the configurations for the network/ training architecture
        :return conv_config: the configs for the conv layers 
        """
        conv_config = {} 
        net_config = {}
        fc_config = []
        

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
                
        return net_config, conv_config
        



    def CONV(self):
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
                    yolo_conv.append(nn.Conv2d(in_channels=3, out_channels=out_filters, kernel_size=kernel, stride=stride, padding=pad, dtype=dtype))
                    yolo_conv.append(nn.BatchNorm2d(num_features=1))
                    yolo_conv.append(nn.LeakyReLU())
                    next_in_filters = out_filters

                else: 
                    yolo_conv.append(nn.Conv2d(in_channels=next_in_filters, out_channels=out_filters, kernel_size=kernel, stride=stride, padding=pad, dtype=dtype))
                    yolo_conv.append(nn.BatchNorm2d(num_features=1))
                    yolo_conv.append(nn.LeakyReLU())
                    next_in_filters = out_filters

            if layer[:4] == "maxp":
                size = conv_configs[layer]["size"]
                stride = conv_configs[layer]["stride"]

                yolo_conv.append(nn.MaxPool2d(kernel_size=size, stride=stride))

        return yolo_conv



if __name__ == "__main__":
    """
    Redmon et al. configs
    """
    PATH = "/home/agastya123/PycharmProjects/ComputerVision/PennFudan/yolov1.cfg"

    model = YOLOv1(config_path = PATH, in_channels=3, split_size=7, num_boxes=2, num_classes=20)
    
    yolo_conv = model.CONV()
    print(yolo_conv)
    
    
    





    

    



    


    
    



    