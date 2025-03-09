# Audio Deepfake Detection 

## Overview

This study explores the effectiveness of different deep learning architectures — CNNs, RNNs, and Transformers — for detecting AI-generated speech. We aim to inform on the best-performing model architecture for deepfake audio detection and contribute insights into developing robust audio deepfake detection models.

## Architectures
- RawNet2, the ResNet family, the EfficientNet family, and custom developed convolution nets will be explored.
- LSTMs will be tested to determine their ability to capture long-term dependencies in speech patterns.
- Wav2vec 2.0 will be fine-tuned.

## Experiment Setup
1.  Models will be trained on the FoR dataset and then benchmarked against theITW dataset to evaluate how well training on generic synthetic
audio data generalizes to detecting specific spoofed audio samples.
2.  The same architectures will be trained and evaluated on the ITW dataset to draw a comparison.

## Datasets

- Fake-Or-Real (FoR) [https://bil.eecs.yorku.ca/wp-content/uploads/2020/01/FoR-Dataset_RR_VT_final.pdf]
- In-The-Wild (ITW) [https://arxiv.org/abs/2203.16263]

## Data Preprocessing

## Hyperparameter Tuning

## Model Performance


