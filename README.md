# Semantic Image Segmentation with Pyramid Atrous Convolution and Boundary-aware Loss
[DeepLabV3](https://arxiv.org/pdf/1706.05587.pdf), 
[This Paper](https://github.com/Tamuel/TF_SemanticSegmentation/blob/master/Semantic%20Image%20Segmentation%20with%20Pyramid%20Atrous%20Convolution%20and%20Boundary-aware%20Loss%20Paper%20(Dongkyu%20Yu%20_%20POSTECH).pdf)

## Overview
Semantic image segmentation network which inspired by Google **DeepLabV3**. We use the ResNet as backbone network for high quality feature extraction. And we design the **Pyramid Atrous Convolution (PAC)** module which employ atrous convolution with multi atrous rate which use same filters. It makes not only the network robust to multiple scales but also reduce the number of parameters for filters. In network learning phase, we apply **Boundary-Aware Loss (BAL)** which can make network focus on hard region of input image like **Hard Example Mining** in object detection area.

## Performance
|**DeepLabV3**||
|--|--|
|Output Stride|16|
|Multi-Grid|1, 2, 4|
|ASPP|6, 12, 18|
|Image Pooling|True|

| Network | mIoU |
|---|---|
| DeepLabV3 (Paper) | 77.21 |
| DeepLabV3 (Regenerated, ResNet101 V2) | 76.68 |
| ResNet101 V2 + PAC | 76.97 |
| ResNet101 V2 + BAL | 77.64 |
| ResNet101 V2 + PAC + BAL | 77.93 |
| **ResNet101 V1 + PAC + BAL** | **78.07** |

## PAC module
![PAC](https://github.com/Tamuel/TF_SemanticSegmentation/blob/master/assets/Pyramid%20atrous%20convolution%20module.png)

## BAL
![BAL](https://github.com/Tamuel/TF_SemanticSegmentation/blob/master/assets/Gaussian%20edge.png)

## Results
![result_image](https://github.com/Tamuel/TF_SemanticSegmentation/blob/master/assets/results.png)

## Prerequisite
* ```Python 3.5.x or 3.6.x```
* ```Tensorflow 1.3 or higher```
* ```Numpy, CV2, PIL, etc```

## Network checkpoints
* [model](https://drive.google.com/open?id=1YFcbw-5nL33Ii9Zm81n2MQLSml1Gu3M0) (ResNet101 V1 + PAC + BAL)
* [resnet_v1_101_beta](https://drive.google.com/open?id=1-jSkSjQcAYBfJX_imShkNtQI6sUyuX-s)

## Usage
First, you have to download checkpoints of ```model``` and unzip it into model directory.
* Run ```_init_.py``` by Python to train network. You can modify several options in ```_init_.py```. You can change number of GPUs, Model directory, Dataset directory, etc by modify flags options.

```
  flags = tf.app.flags

  FLAGS = flags.FLAGS

  flags.DEFINE_integer('num_gpu', 1,
                       'Number of GPUs to use.')

  flags.DEFINE_string('base_architecture', 'resnet_v1_101',
                      'The architecture of base Resnet building block.')

  flags.DEFINE_string('pre_trained_model',
                      './init_checkpoints/' + FLAGS.base_architecture + '/model.ckpt',
                      'The architecture of base Resnet building block.')

  flags.DEFINE_string('model_dir', './model',
                      'Base directory for the model')

  flags.DEFINE_string('data_dir', './dataset/',
                      'Path to the directory containing the PASCAL VOC data tf record.')
  ...
```

* Run ```evaluate.py``` by Python to evaluate network.
* Run ```prediction.py``` by Python to evaluate prediction. You can modify ```model_dir```, ```input_dir```, ```output_dir``` to modify directories to predictions.

```
  model_dir = './test_model'
  input_dir = './test_input'
  output_dir = './test_output'
```
