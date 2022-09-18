"""
    Defining the different object detection models 
"""
import torch
import torch.nn as nn
import numpy as np
torch.manual_seed(0)
import configparser

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
        """
        conv_config = []
        pool_config = [] 
        gen_config = []
        fc_config = []
        
        for sect in self.config.sections():
            if len(self.config.options(sect)) == 6: 
                conv_config.append((sect, self.config.options(sect)))
            elif len(self.config.options(sect)) == 2:
                pool_config.append((sect, self.config.options(sect)))
            else: 
                gen_config.append((sect, self.config.options(sect)))

        return gen_config


        

    def create_layers(self):
        """Method to create convolutional layers from the configs"""
        pass 


    def network(self):
        """
        Method to actually define the YOLO network.
        """
        pass




        


        

        
        






if __name__ == "__main__":
    """
    Redmon et al. configs
    """
    PATH = "/home/agastya123/PycharmProjects/ComputerVision/ChessPieces/yolov1.cfg"

    model = YOLOv1(config_path = PATH, in_channels=3, split_size=7, num_boxes=2, num_classes=20)
    print(model.configurations())



    

    



    


    
    



    