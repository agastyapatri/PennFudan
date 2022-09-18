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
        Method to load and sort the configuratoins in self.config
        """
        conv_configs = []
        gen_configs = []
        fc_configs = []
        
        for sect in self.config.sections():
            items = self.config.items(sect)
            if len(items) == 2 or len(items) == 6: 
                conv_configs.append(( sect, items) )

            else:
                gen_configs.append((sect, items))



        return gen_configs
        

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
    PATH = "/home/agastya123/PycharmProjects/ComputerVision/ChessPieces/figures-results/yolov1.cfg"

    model = YOLOv1(config_path = PATH, in_channels=3, split_size=7, num_boxes=2, num_classes=20)
    
    print(model.configurations())
    


    
    



    