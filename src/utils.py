import torch 
import torch.nn as nn 

class Utilities():
        # Class to define some helper functions
    
    def IoU(self, box1, box2 ):
        """
            Intersection over union
        """
        image1, (H1, W1), targets1 = box1
        image2, (H2, W2), targets2 = box2 


        pass 

    def NMS(self):
        """
            Non Max Suppression 
        """
        pass 

    def MAP(self):
        """
            Mean Average Precision
        """
        pass 