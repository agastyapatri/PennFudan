# **PennFudan Object Detection**
_This project is intended for me to cut my teeth on object detection algorithms. The focus of this project will be the implementation of essential algorithms such as:_

    1. YOLOv1
    2. R-CNN and its derivatives
    3. Single Shot Multibox Detection

_to detect pedestrians, using the Penn-Fudan pedestrian detection and segmentation dataset_ \[1\]

## **Introduction**
The images are taken from scenes around campuses. Each image will have at least one pedestrian in it.

The heights of labeled pedestrians in this database fall into \[180,390 \] pixels. All labeled pedestrians are straight up.

There are 170 images with 345 labeled pedestrians, among which 96 images are taken from around University of Pennsylvania, and other 74 are taken from around Fudan University.

_This description is taken verbatim from \[ 1 \]_

The annotations for the images use the `PASCAL Annotation version 1.00`

## **Models**
_This section details the different models that are  built + trained_

* **YOLOv1** 
  The labels for YOLO take the form :
    $$ label_{cell} = [\{c_n\}, p_{c}, x,y, w, h]$$
    Where n is the number of classes, $p_c$ is the probability that there is an object $\{x, y, w, h\}$ are the dimensions of the bounding boxes.

    The target shape of the output for one image is $(S, S, 7)$
    prediction shape for one image is $(S, S, )$



## **Modus Operandi**



## **Results**


## **References**

\[1\]: [Penn-Fudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/)



\[2\]: [TorchVision Object Detection Tutorials](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)




1. https://blog.roboflow.com/object-detection/#computer-vision-workflow
2. https://blog.roboflow.com/how-to-train-yolov6-on-a-custom-dataset/
3. 



