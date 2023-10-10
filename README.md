# Scrub Typhus Cell Counting techniques

## Objective of this project
Explore deep learning techniques to improve the speed and accuracy of scrub typhus cell counting by adjusting backbone size and transitioning from instance segmentation to object detection.

### Deep learning models 
The proposed deep learning models were created using MMDetection, a software library that offers a diverse range of deep learning models. You can access it at https://github.com/open-mmlab/mmdetection/projects. Four intriguing models were utilized in this project: Mask R-CNN for instance segmentation, and Faster R-CNN, Adaptive Training Sample Selection (ATSS), and You Only Look Once version 3 (YOLOv3) for object detection.

### Image processing technique
CellProfiler is a notable tool for processing biological images, with an example illustrating its application in cell counting in the 'Output_Cellprofiler' directory.


# Image Dataset
## What is Scrub Typhus?
Scrub typhus, caused by the bacterium Orientia tsutsugamushi, is transmitted by infected mites in vegetation-rich areas of Asia, the Pacific Islands, and the Middle East. Symptoms include fever, headache, rash, and swollen lymph nodes. Diagnosis relies on clinical and lab tests, with early antibiotic treatment being vital. No vaccine is widely available, so preventive measures, like protective clothing and insect repellent use, are key. Efforts continue to raise awareness and improve diagnostics and treatment in endemic regions. Timely medical attention is critical for recovery.

## Image character
All organelles were fluorescently stained, with red for cell boundaries, blue for nucleus boundaries, and green for bacteria boundaries, and then captured using high-content screening.
The dataset includes numerous images containing both control genes and knockdown genes. It's worth noting that only the images with enhanced quality for the first control gene have been uploaded in Enhanced_BioImage directory. In contrast, the Image_Datasets directory has combined three fluorescence images into a single image for streamlined integration into deep learning models.

## Data annotation
In-house software for data annotation has been generated and is available at https://github.com/Chuenchat/cellLabel


# How to Run Training and Evaluation using mmdetection
### Prerequisites
Before you begin, ensure you have mmdetection installed. If not, you can follow the installation instructions from mmdetection's official repository.<br>
Pre-trained models can be downloaded from the official mmdetection website before you starting.<br>
Several files in the mmdetection repository were modified (looking at modified_mmdetection folder) and utilized to evaluate metrics for our project.<br>

### Image Dataset
Images of the control gene of the first plate were uploaded in here.<br>
1. Enhaned_BioImage folder: Three fluorescence channels (Red for cell, Blue for nucleus, and Green for bacteria) were enhanced image quality.<br>
2. Image_Datasets folder: Merge enhaned three fluorescence images into an image and it is transfered into objective models<br>


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

