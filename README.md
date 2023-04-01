# 人工智慧醫學影像專題列表

The following list are the project of the course I have worked with.

## CXR Image Classification

Is this apporach, we use the cxr data for classify a cxr image is normal one or pnuemonia,
A simple approach has been stored in the folder of cxr-image-classification

### Quick Overview

- Difficulity: Imbalance data
- Test Acc: 90.4 %
- Platform: W & B
- GPU: 3050 4GM
- Epoch: 5 + 1 (5 for classifier, 1 for all paramaters)
- Loss: WeightedBCE

### Report Location

Here is the report generated from WandB platform

https://api.wandb.ai/links/ddcvlab/dqcal3k2

## EEG Classification

Is this apporach, we use the eeg data for classify a eeg signal for binary classification,
A simple approach has been stored in the folder of eeg-classification

### Quick Overview

- Difficulity: Overfitting
- Test Acc: 87.22 %
- Platform: W & B
- GPU: P100 16GB
- Epoch: 5000 
- Loss: CrossEntropy
