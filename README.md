## Object Detection and Data annotation for Scrub Typhus Cell Counting


<summary>Object Detection part</summary>

In a case study focused on high-content RNAi screening for Orientia tsutsugamushi bacterium infection, we utilized MMDetection, a software library with a wide variety of object detection models and can be found in https://github.com/open-mmlab/mmdetection/projects), to perform cell counting. 

Two-stage detectors like Faster RCNN, Mask RCNN, and Cascade Mask RCNN are known for their accuracy but they are slower in terms of inference speed. Therefore, we proposes an alternative approach that involves decreasing backbone size with the two-stage detectors, and also exploring the use of newer one-stage detectors such as Adaptive Training Sample Selection (ATSS) and You Only Look Once version 3 (YoloV3).

By using these different approaches, we were able to create ten different models for cell counting. This suggests that there are multiple ways to approach the problem of cell counting using object detection models, and different models may be suitable for different applications depending on the specific requirements for accuracy and inference speed.

For testing:
1. Pre-trained models can be downloaded from the official mmdetection website.
2. Images of the control gene of the first plate were uploaded in here.
      
      Enhaned_BioImage: Three fluorescence channels (Red for cell, Blue for nucleus, and Green for bacteria) were enhanced image quality.
      Image_Datasets: Merge enhaned three fluorescence images into an image and it is transfered into objective models
3. Coco files cannot uploded because it's huge.
4. Please follow each command step: 1_Trainmodel.sh, 2_Testmodels.sh, 3_ConfusionMatrix.sh, and 4_CalTrainTime.sh. 

     These command steps have been modified to be suitable for our project.<p style="margin-bottom: 40px;"></p><br>


<summary>Data annotation</summary>
We developed in-house software for data annotation part that can be found in https://github.com/Chuenchat/cellLabel.
<p style="margin-bottom: 40px;"></p><br>


<summary>Image processing technique</summary>
We updated an example of image processing technique in Output_Cellprofiler folder.
