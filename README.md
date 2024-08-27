# DSHANet

# Introduction
Cropland is crucial for national food security, maintaining agricultural product quality, and ensuring environmental safety. Consequently, there is an urgent need for cropland change detection (CD) using high-resolution remote sensing images to accurately track cropland distribution and changes. However, the irregular shapes of cultivated areas and challenges in feature fusion complicate boundary localization, posing a risk of losing critical change features. To address these issues, we introduce the dynamic sparse hierarchical attention-driven cropland change detection network (DSHANet), which combines a vision transformer with dynamic sparse hierarchical attention (DSHA-Former) and holistic complementation fusion (HCF) modules. DSHA-Former effectively detects targets and refines edge features in images of various sizes. Concurrently, HCF preserves key details by integrating core, setting, margin, and panorama data, supplementing global image-level content comprehensively. This significantly improves the definition of changed areas in the process of merging features from different scales, thus enhancing cropland CD. We evaluate our approach on the CL-CD dataset, achieving an F1-score of 79.49%. In addition, the network demonstrates strong generalization capabilities on both LEVIR-CD and WHU-CD datasets, with F1-scores of 92.42% and 92.18%, respectively.
![Fig 1](https://github.com/user-attachments/assets/843f0832-e5fb-4617-aeb8-c1f26900b941)

# Datasets 
CL-CD  https://github.com/liumency/CropLand-CD  

WHU-CD  http://gpcv.whu.edu.cn/data/  

LEVIR-CD  https://justchenhao.github.io/LEVIR/ 


# Train 
1. Download the dataset.
2. Modify paths of datasets, then run train.py.

# Test
Modify paths of datasets, then run eval.py.

# Results
## Experimental metrics on the CL-CD dataset.
|                     | **Method**   | **Precision** | **Recall** | **F1-score** | **mIOU**  | **IOU_0** | **IOU_1** | **OA**    | **Kappa** |
| ------------------- | ------------ | ------------- | ---------- | ------------ | --------- | --------- | --------- | --------- | --------- |
| Method   comparison | FC_EF        | 53.3          | 73.51      | 61.8         | 68.78     | 92.84     | 44.71     | 93.24     | 58.19     |
| BITNet              | 57.46        | 77.44         | 65.97      | 71.45        | 93.69     | 49.22     | 94.05     | 62.79     |           |
| HFANet              | 62.66        | **79.13**     | 69.94      | 94.62        | 53.77     | 74.2      | 94.94     | 67.21     |           |
| MSCANet             | 59.14        | 72.23         | 65.03      | 93.89        | 48.18     | 71.04     | 94.22     | 61.91     |           |
| SARASNet            | 71.25        | 75.63         | 73.37      | **95.67**    | 57.95     | **76.81** | 95.92     | 71.16     |           |
| CAGNet              | **73.93**    | 73.12         | **73.52**  | 77.00        | **95.86** | 58.13     | **96.08** | **71.41** |           |
| CSINet              | 64.02        | **82.72**     | 72.18      | **94.94**    | 56.47     | **75.7**  | 95.25     | 69.63     |           |
| Ours                | **85.21**    | 74.49         | **79.49**  | 81.47        | **96.97** | 65.97     | **97.14** | **77.96** |           |
| Ablation            | w/o DSHA+HCF | 66.07         | 64.85      | 65.45        | 71.65     | 94.65     | 48.65     | 94.91     | 62.70     |
| w/o DSHA            | 67.86        | 67.15         | 67.50      | 72.94        | 94.94     | 50.95     | 95.19     | 64.91     |           |
| w/o HCF             | **83.58**    | **74.98**     | **79.05**  | **81.11**    | **96.87** | **65.36** | **97.04** | **77.46** |           |
| DSHANet             | **85.21**    | **74.49**     | **79.49**  | **81.47**    | **96.97** | **65.97** | **97.14** | **77.96** |           |

## Generalization experiment results comparing our method with other SOTA methods on the LEVIR-CD and WHU-CD datasets.
| **Dataset** | **Method** | **Precision** | **Recall** | **F1-score** | **mIOU**  | **IOU_0** | **IOU_1** | **OA**    | **Kappa** |
| ----------- | ---------- | ------------- | ---------- | ------------ | --------- | --------- | --------- | --------- | --------- |
| LEVIR-CD    | FC-EF      | 79.91         | 82.84      | 81.35        | 81.97     | 95.38     | 68.56     | 95.80     | 78.99     |
| BITNet      | 87.32      | 91.41         | 89.32      | 89.00        | **97.31** | 80.70     | 97.59     | 87.96     |           |
| HFANet      | 83.36      | 91.47         | 87.23      | 96.71        | 77.34     | 87.03     | 97.04     | 85.56     |           |
| MSCANet     | 83.75      | 91.85         | 87.61      | 96.81        | 77.95     | 87.38     | 97.13     | 85.99     |           |
| SARASNet    | 89.48      | **92.64**     | 91.03      | **97.75**    | 83.54     | 90.64     | 97.98     | 89.90     |           |
| CAGNet      | **92.63**  | 90.53         | **91.57**  | 91.20        | **97.95** | 84.45     | **98.16** | **90.53** |           |
| CSINet      | 88.61      | **93.61**     | 91.04      | 97.73        | 83.55     | **90.64** | 97.96     | 89.89     |           |
| Ours        | **94.05**  | 90.86         | **92.42**  | **98.17**    | 85.91     | **92.05** | **98.36** | **91.50** |           |
| WHU-CD      | FC-EF      | 70.43         | 92.31      | 79.90        | 82.12     | 97.72     | 66.53     | 97.82     | 78.77     |
| BITNet      | 82.35      | **92.59**     | 87.17      | 87.96        | **98.66** | 77.26     | 98.72     | 86.50     |           |
| HFANet      | 74.19      | 89.56         | 81.15      | 97.96        | 68.29     | 83.12     | 98.04     | 80.13     |           |
| MSCANet     | 83.07      | 90.70         | 86.72      | 98.63        | 76.55     | 87.59     | 98.69     | 86.03     |           |
| SARASNet    | 82.88      | **94.02**     | 88.10      | **98.75**    | 78.73     | **88.74** | 98.81     | 87.47     |           |
| CAGNet      | **92.67**  | 90.33         | **91.48**  | 91.74        | 96.17     | 84.30     | **99.21** | **91.07** |           |
| CSINet      | 86.28      | 91.51         | 88.82      | **98.87**    | 79.88     | **89.38** | 98.92     | 88.25     |           |
| Ours        | **92.97**  | 91.41         | **92.18**  | 92.37        | **99.24** | 85.50     | **99.27** | **91.80** |           |


# Visualization of results
## Visualization of cropland CD and attention map results on the CL-CD dataset
![Fig 3 - 副本-min](https://github.com/user-attachments/assets/a0936a8c-2c12-47a9-9c5c-d44815828a67)


## Visualization of cropland CD ablation results on the CL-CD dataset
![Fig 4](https://github.com/user-attachments/assets/d1975e05-a2d7-4ac2-8c3f-bdf58d804554)

## Box-plot comparison chart of quantitative indicators for DSHANet
![Fig 5](https://github.com/user-attachments/assets/e87b5bc0-a8fc-4716-b269-9e2d9e6e0c74)


## Visualization of experimental results on other datasets
![Fig 6](https://github.com/user-attachments/assets/fc78ebc8-7bd6-4159-ba30-0084e3f74ed2)







