# **Object Detection**
_This project is an implementation of the key ideas that enable Object Detection. There are a few standard algorithms which will be tested. I've started this endeavour as an Object Detection counterpart to the purpose that the  MNIST Image Classification task serves. Any further experimentation with object detection algorithms will be done here._


## **Introduction**
I will be focusing on the implementation of the members of the families of three main Object Detection Algorithms:
1. Single Shot Multibox Detectors (SSD)
2. YOLOv3
3. Fast(er) R-CNN (FRCNN)

The /figures-results diretory also contains the config files which have the parameters for the network. 

### _**Single Shot Multibox Detection**_
The model consists of a base network followed by several multiscale feature map blocks. The base network extracts features from the image and for this purpose, VGG16 is used [3].





## **Process**




## **Results**

## **References** 
To be edited in proper citation etiquette. 

[1] YOLOv3 paper, Redmon et al.

[2] Single Shot Multibox Detector Paper, Liu et al
[3] Faster R-CNN paper, 

1. https://blog.roboflow.com/object-detection/#computer-vision-workflow
2. https://blog.roboflow.com/how-to-train-yolov6-on-a-custom-dataset/
3. 




## CONV
equential(
  (0): CNNBlock(
    (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (batchnorm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (2): CNNBlock(
    (conv): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (4): CNNBlock(
    (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (5): CNNBlock(
    (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (6): CNNBlock(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (7): CNNBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (8): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (9): CNNBlock(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (10): CNNBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=Truequential(
  (0): CNNBlock(
    (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (batchnorm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (1): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (2): CNNBlock(
    (conv): Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (3): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (4): CNNBlock(
    (conv): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (5): CNNBlock(
    (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (6): CNNBlock(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (7): CNNBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (8): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (9): CNNBlock(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (10): CNNBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (11): CNNBlock(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (12): CNNBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (13): CNNBlock(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (14): CNNBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (15): CNNBlock(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (16): CNNBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (17): CNNBlock(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (18): CNNBlock(
    (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (19): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (20): CNNBlock(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (21): CNNBlock(
    (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (22): CNNBlock(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (23): CNNBlock(
    (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (24): CNNBlock(
    (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (25): CNNBlock(
    (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (26): CNNBlock(
    (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (27): CNNBlock(
    (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
)e)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (11): CNNBlock(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (12): CNNBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (13): CNNBlock(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (14): CNNBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (15): CNNBlock(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (16): CNNBlock(
    (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (17): CNNBlock(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (18): CNNBlock(
    (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (19): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
  (20): CNNBlock(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (21): CNNBlock(
    (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (22): CNNBlock(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (23): CNNBlock(
    (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (24): CNNBlock(
    (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (25): CNNBlock(
    (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (26): CNNBlock(
    (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
  (27): CNNBlock(
    (conv): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (batchnorm): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (leakyrelu): LeakyReLU(negative_slope=0.1)
  )
)