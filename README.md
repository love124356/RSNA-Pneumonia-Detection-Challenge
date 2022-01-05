# RSNA-Pneumonia-Detection-Challenge

This repository gathers the code for RSNA Pneumonia Detection Challenge from the [Kaggle competition](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview).

We use [Faster R-CNN](https://www.kaggle.com/anastasiiaselezen/rsna-pneumonia-detection-challenge)[1], an open competition notebook using Faster R-CNN to implement.

## Reproducing Submission
We need to do some pre-preparation for training and testing on the stage 2 dataset.

To reproduce my submission without retrainig, do the following steps:
1. [Requirement](#Requirement)
2. [Repository Structure](#Repository-Structure)
3. [Inference](#Inference)

## Hardware

Ubuntu 18.04.5 LTS

Intel® Core™ i7-3770 CPU @ 3.40GHz × 8

GeForce GTX 1080/PCIe/SSE2

## Requirement
All requirements should be detailed in requirements.txt.

```env
$ cd RSNA-Pneumonia-Detection-Challenge
$ conda env create -f environament.yml
```
We use VSCode to open train.ipynb or inference.ipynb, and choose the kernel name "rsnatest". Next, Run all the cells.

Maybe you can use Kaggle or Colab to run these codes.

The jpg images can be downloaded from [here](https://www.kaggle.com/sovitrath/rsna-pneumonia-detection-2018?select=input) and put in root.
The origin .dam can be downloaded from [here](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data) and put in root.

## Repository Structure

The repository structure is:
```
RSNA-Pneumonia-Detection-Challenge(root)
  +-- input                         # all files need in this task
  |   +-- images                    # training data
  |   +-- samples                   # testing data
  |   +-- stage_2_train_labels.csv  # training labels
  +-- stage_2_test_images           # testing .dcm 
  +-- models                        # model weights
  +-- calculate_map.py              # ensemble utils
  +-- ensemble.py                   # reproduce my submission file
  +-- inference.ipynb               # for testing 
  +-- train.ipynb                   # for training model
  +-- environment.yml               # yaml file for establishing the environment
```

## Training

To train the model, run train.ipynb:

The "FOLD" parameter is for cross validation. Please modify the number from 0~4 to train.

Trained model will be saved as ```models/fasterrcnn_resnet50_fpn_pneumonia_detection_best.pth```

## Inference

Please download [these five model weights](https://reurl.cc/nE0RDn) if you want to reproduce my submission file, and put them in root or the folder you create.

To reproduce my submission file or test the model you trained, run inference.ipynb.

Note that you need to modify MODEL_PATH and the SUBMISSION_PATH for five model weights(SUBMISSION_PATH should not the same for different weights).

Prediction file will be saved as ```root/{SUBMISSION_PATH}```


Finally, you can run the following command to ensemble the five .csv (using conda env "rsnatest")[2].

We also provide the [five .csv](https://reurl.cc/dX5v7q).

```py
python ensemble.py stage_2_test_images output ensemble.csv 0.csv 1.csv 2.csv 3.csv 4.csv
```

```0.csv 1.csv 2.csv 3.csv 4.csv``` is the csv you create above step.

## Reference
[1] [Faster R-CNN notebook](https://www.kaggle.com/anastasiiaselezen/rsna-pneumonia-detection-challenge)

[2] [Ensemble Method](https://gist.github.com/raytroop/abbfb31772a5c8797dade81193da16d5)
