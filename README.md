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

# Visualization of results
## Visualization of cropland CD and attention map results on the CL-CD dataset
![Fig 3 - 副本-min](https://github.com/user-attachments/assets/a0936a8c-2c12-47a9-9c5c-d44815828a67)


## Visualization of cropland CD ablation results on the CL-CD dataset
![Fig 4](https://github.com/user-attachments/assets/d1975e05-a2d7-4ac2-8c3f-bdf58d804554)

## Box-plot comparison chart of quantitative indicators for DSHANet
![Fig 5](https://github.com/user-attachments/assets/e87b5bc0-a8fc-4716-b269-9e2d9e6e0c74)


## Visualization of experimental results on other datasets
![Fig 6](https://github.com/user-attachments/assets/fc78ebc8-7bd6-4159-ba30-0084e3f74ed2)







