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
        """
        conv_config = {}
        pool_config = {} 
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
                pool_config[sect] =  {
                    "size" : self.config.get(section=sect, option="size"), 
                    "stride": self.config.get(section=sect, option="stride"),
                    }


            # Network Hyperparameter Configurations
            elif sect[:3] == "net":
                for option in self.config.options(sect):
                    net_config[option] = self.config.get(section=sect, option=option)
                
        
        return net_config
        




    def create_conv_layer(self):
        configs = self.configurations()


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
    PATH = "/home/agastya123/PycharmProjects/ComputerVision/PennFudan/yolov1.cfg"

    model = YOLOv1(config_path = PATH, in_channels=3, split_size=7, num_boxes=2, num_classes=20)
    
    print(model.configurations())




    

    



    


    
    



    