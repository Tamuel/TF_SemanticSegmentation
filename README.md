# Semantic Image Segmentation with Pyramid Atrous Convolution and Boundary-aware Loss

[Paper](https://github.com/Tamuel/TF_SemanticSegmentation/blob/master/Semantic%20Image%20Segmentation%20with%20Pyramid%20Atrous%20Convolution%20and%20Boundary-aware%20Loss%20Paper%20(Dongkyu%20Yu%20_%20POSTECH).pdf)

## Overview
Semantic image segmentation network which inspired by Google **DeepLabV3**. We use the ResNet as backbone network for high quailty feature extraction. And we design the **Pyramid Atrous Convolution** module which employ atrous convolution with multi atrous rate which use same filters. It makes not only the network robust to multiple scales but also reduce the number of parameters for filters. In network learning phase, we apply **Boundary-Aware Loss** which can make network focus on hard region of input image like **Hard Example Mining** in object detection area.

## Performance
| Network | mIoU |
|---|---|
|DeepLabV3 (Paper) | 77.21 |
|DeepLabV3 (Regenerated) | 76.68 |
|ResNet101 + PAC | 76.97 |
|DeepLabV3 + BAL | 77.64 |
|DeepLabV3 + PAC + BAL | 77.93 |

## PAC module & BAL
![PAC](https://github.com/Tamuel/TF_SemanticSegmentation/blob/master/assets/Pyramid%20atrous%20convolution%20module.png)

![BAL](https://github.com/Tamuel/TF_SemanticSegmentation/blob/master/assets/Gaussian%20edge.png)


## Results
![result_image](https://github.com/Tamuel/TF_SemanticSegmentation/blob/master/assets/results.png)
