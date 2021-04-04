# COVID-19 Cough Classification using PyTorch

## Overview

The aim of this project is to classify audio recordings of coughs into COVID-19 positive and negative.

## Installation

### Install required libraries

```
pip install -r requirements.txt
```

## Running the code

The code is split into four notebooks:

1. [001_prepare_data.ipynb](001_prepare_data.ipynb): This notebook extracts the features of the wav files and writes them to a new csv file.
2. [002_training.ipynb](002_training.ipynb): Trains and evaluates the CoughNet using the extracted features. Finally a checkpoint is saved.
3. [003_inference.ipynb](003_inference.ipynb): Uses the saved checkpoint to predict on an input wav file. There is also a [pretrained checkpoint](checkpoints/) available.
4. [004_k_fold_cross_validation.ipynb](004_k_fold_cross_validation.ipynb): K-fold Cross Validation for objective evaluation.

## Datasets

### Kaggle Cough-Classifier Dataset

https://www.kaggle.com/himanshu007121/coughclassifier-trial

The major problem with this dataset is, that it is highly unbalanced. Only 19 of the 170 examples are labeled as positive. <br/>
Therefore, in addition to the good test accuracy, the model shows a relatively high false-negative rate. <br/>
More data, especially positive examples, is needed to develop a reliable, cough audio-based Covid-19 test. <br/>

### Virufy Dataset

https://github.com/virufy/virufy_data/tree/main/clinical/segmented

This dataset is provided by the developers of the Virufy app. They offer a webservice (https://virufy.org/en/) which can detect a COVID-19 signature in recordings using an AI algorithm. The open dataset on github contains 121 examples of which 48 are labeled as positive.

### Balanced Dataset

The balanced dataset is created by combining the Kaggle and the Virufy dataset. Since they both contain more negative than positive examples, downsampling is used to create a perfectly balanced dataset.

### Other Datasets

Other datasets have not been investigated, yet.

- **Coswara**: https://github.com/iiscleap/Coswara-Data
- **COUGHVID**: https://zenodo.org/record/4048312

## The Model

The model used is a simple DNN with 6 Layers. The input is a 26 values feature vector calculated from the audio files using librosa.

![Model][img/model.png]

## Results

The balanced dataset is split into 8 folds and evaluated using cross validation.<br/>
The model converges after 20 epochs of training.

|             | Train Accuracy | Test Accuracy |
| ----------- | -------------: | ------------: |
| Fold 0      |       100.00 % |       94.12 % |
| Fold 1      |       100.00 % |       88.24 % |
| Fold 2      |       100.00 % |       94.12 % |
| Fold 3      |       100.00 % |       76.47 % |
| Fold 4      |       100.00 % |       88.24 % |
| Fold 5      |       100.00 % |       94.12 % |
| Fold 6      |       100.00 % |      100.00 % |
| Fold 7      |        96.61 % |       87.50 % |
| **Average** |    **99.58 %** |   **90.35 %** |

![Confusion Matrix][img/confusion_matrix_balanced.png]

## Credits

The Model is based on the Keras notebook by Himanshu which can be found here:
https://www.kaggle.com/himanshu007121/covid-19-cough-classification
