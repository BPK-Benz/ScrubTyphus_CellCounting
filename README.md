## Object Detection and Data annotation for Scrub Typhus Cell Counting


### Object Detection part

In a case study focused on high-content RNAi screening for Orientia tsutsugamushi bacterium infection, we utilized MMDetection, a software library with a wide variety of object detection models and can be found in https://github.com/open-mmlab/mmdetection/projects), to perform cell counting. 

Two-stage detectors like Faster RCNN, Mask RCNN, and Cascade Mask RCNN are known for their accuracy but they are slower in terms of inference speed. Therefore, we proposes an alternative approach that involves decreasing backbone size with the two-stage detectors, and also exploring the use of newer one-stage detectors such as Adaptive Training Sample Selection (ATSS) and You Only Look Once version 3 (YoloV3).

By using these different approaches, we were able to create ten different models for cell counting. This suggests that there are multiple ways to approach the problem of cell counting using object detection models, and different models may be suitable for different applications depending on the specific requirements for accuracy and inference speed.<p style="margin-bottom: 40px;"></p><br>


### Data annotation
We developed in-house software for data annotation part that can be found in https://github.com/Chuenchat/cellLabel.
<p style="margin-bottom: 40px;"></p><br>


### Image processing technique
We updated an example of image processing technique in Output_Cellprofiler folder.<br><br>


# How to Run Training and Evaluation using mmdetection
### Prerequisites
Before you begin, ensure you have mmdetection installed. If not, you can follow the installation instructions from mmdetection's official repository.<br>
Pre-trained models can be downloaded from the official mmdetection website before you starting.<br>
Several files in the mmdetection repository were modified (modified_mmdetection) and utilized to evaluate metrics for our project.<br>

### Image Dataset
Images of the control gene of the first plate were uploaded in here.<br>
1. Enhaned_BioImage: Three fluorescence channels (Red for cell, Blue for nucleus, and Green for bacteria) were enhanced image quality.<br>
2. Image_Datasets: Merge enhaned three fluorescence images into an image and it is transfered into objective models<br>


### Training
Navigate to the card_det_models_configs directory.<br>
Use the following command to initiate training:
python tools/train.py <path_to_config_file><br>
Replace <path_to_config_file> with the path to the desired model config file.<br>
Seeing example commands in 1_Trainmodel.sh.<br>

### Evaluation: Testing, Confusion matrix, Train_Time
Use the following command for testing:
python tools/test.py <path_to_config_file> <path_to_checkpoint> --show-dir<path_to_results> --eval bbox --out <path_to_pkl_file> --eval-option proposal_nums="(200,300,1000)" classwise=True save_path=<path_to_save><br>
Seeing example commands in 2_Testmodels.sh.<br>


Use the following command for confusion matrix:
python tools/analysis_tools/confusion_matrix.py <path_to_config_file>   <path_to_pkl_file> <path_to_save><br>
Seeing example command in 3_ConfusionMatrix.sh<br>


Use the following command for train_time:
python tools/analysis_tools/analyze_logs.py cal_train_time log.json
Seeing example command in 4_CalTrainTime.sh. 

