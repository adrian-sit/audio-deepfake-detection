# Audio Deepfake Detection 

## Overview

This study explores the effectiveness of different deep learning architectures — CNNs and RNNs — for detecting AI-generated speech. We aim to inform on the best-performing model architecture for deepfake audio detection and contribute insights into developing robust audio deepfake detection models.

## Architectures
- RawNet2 [https://arxiv.org/abs/2011.01108], the ResNet family, the EfficientNet family is implemented.
- LSTMs is tested to determine their ability to capture long-term dependencies in speech patterns.

## Datasets

- Fake-Or-Real (FoR) [https://bil.eecs.yorku.ca/wp-content/uploads/2020/01/FoR-Dataset_RR_VT_final.pdf]
  - Real and generic TTS
  - Class balanced
  - Re-recorded to simulate transmission through a voice channel (for our experiments this will simulate noise)
- In-The-Wild (ITW) [https://arxiv.org/abs/2203.16263]
  - Real and deepfaked celebrities and politicians

## Feature Representations

- ResNet, EfficientNet, LSTMs: Spectrograms
  - cqtspec
  - logspec
  - melspec
- RawNet2: Raw audio

## Data Preprocessing

Convolutional neural network (CNN) style models expect input of a certain shape, with performance being reduced when there is a significant dimension mismatch. The audio is cropped to 2 seconds, and all spectral features are resized when using CNNs. 

## Experiment Setup
1.  Models is trained on the FoR dataset and then benchmarked against the ITW dataset to evaluate how well training on generic synthetic audio data generalizes to detecting specific spoofed audio samples.
2.  Transfer learning is done from the model trained on the FoR dataset to the ITW dataset.
3.  The same architectures will be trained and evaluated on the ITW dataset from scratch to draw a comparison.

## Evaluation Metrics

This work uses the equal error rate (EER) alongside accuracy. After training, models were timed on how long inference takes.

## Model Performance

We experimented with different numbers of epochs and patience levels for early stopping, reporting the best results. We report the test set accuracy, EER, and best validation accuracy for the FoR dataset. Then, we report the initial EER and accuracy of the FoR model evaluated on the entire ITW dataset to evaluate generalization. We then report the test set accuracy and EER after doing transfer learning from the FoR model on to the ITW dataset. Finally, we report the results of training purely on the ITW dataset.

| Model          | Features      | FoR Test EER % | FoR Test Accuracy | FoR Best Validation Accuracy | ITW Initial EER % | ITW Initial Test Accuracy | ITW Transfer Learning EER % | ITW Transfer Learning Test Accuracy | ITW Pure Training EER % | ITW Pure Training Test Accuracy | ITW Inference Time (ms) |
|----------------|---------------|------------|----------------|---------------------------|----------------|------------------------|---------------------------|-------------------------------|----------------------|-----------------------------|----------------------|
| LSTM           | cqt spec      | 20.098     | 79.044         | 94.429                    | 57.699         | 57.110                 | 0.25182                   | 99.717                        | 0.46156              | 99.623                      | 0.082971             |
| LSTM           | log spec      | 43.750     | 56.005         | 84.135                    | 58.192         | 55.037                 | 2.03510                   | 97.200                        | 2.79990              | 96.793                      | 0.140880             |
| LSTM           | mel spec      | 39.842     | 61.272         | 57.931                    | 55.803         | 55.803                 | 0.74340                   | 99.248                        | 0.08361              | 99.248                      | 0.086310             |
| EfficientNet-b0| cqt spec      | 22.058     | 77.574         | 96.602                    | 30.410         | 70.839                 | 0.22916                   | 99.771                        | 0.27180              | 99.697                      | 0.978320             |
| EfficientNet-b0| log spec      | 29.044     | 64.216         | 97.192                    | 31.173         | 67.734                 | 0.34716                   | 99.623                        | 0.37265              | 99.355                      | 1.046600             |
| EfficientNet-b0| mel spec      | 35.241     | 59.738         | 93.366                    | 41.152         | 55.623                 | 0.60723                   | 99.127                        | 0.60723              | 99.127                      | 1.048400             |
| EfficientNet-b1| cqt spec      | 26.225     | 71.936         | 98.003                    | 30.755         | 71.037                 | 0.20431                   | 99.777                        | 0.31689              | 99.702                      | 22.850000            |
| EfficientNet-b1| log spec      | 31.712     | 63.418         | 97.721                    | 35.409         | 62.568                 | 0.37419                   | 99.683                        | 0.37419              | 99.641                      | 23.273000            |
| EfficientNet-b1| mel spec      | 45.267     | 53.721         | 97.566                    | 52.067         | 49.066                 | 0.39235                   | 99.506                        | 0.51887              | 99.523                      | 23.014000            |
| ResNet50       | cqt spec      | 17.034     | 78.186         | 98.128                    | 26.247         | 73.615                 | 0.27771                   | 99.748                        | 0.51251              | 99.685                      | 4.057500             |
| ResNet50       | log spec      | 29.289     | 63.859         | 98.309                    | 29.803         | 63.019                 | 0.34512                   | 99.600                        | 0.49853              | 99.354                      | 3.470400             |
| ResNet50       | mel spec      | 33.013     | 62.655         | 96.473                    | 33.104         | 59.653                 | 0.45761                   | 99.492                        | 0.56319              | 99.360                      | 3.104300             |
| ResNet18       | cqt spec      | 16.073     | 77.053         | 98.707                    | 25.312         | 73.447                 | 0.47170                   | 99.602                        | 0.62731              | 99.462                      | 3.088400             |
| ResNet18       | log spec      | 28.002     | 66.853         | 98.340                    | 34.062         | 71.253                 | 0.56960                   | 99.338                        | 0.71953              | 99.334                      | 3.092300             |
| ResNet18       | mel spec      | 29.177     | 70.066         | 97.043                    | 40.211         | 70.260                 | 0.57260                   | 99.334                        | 0.77510              | 99.301                      | 3.119100             |
| RawNet2        | raw waveform | 8.5282     | 89.731         | 99.145                    | 12.253         | 85.728                 | 0.09627                   | 99.923                        | 0.12709              | 99.861                      | 31.619000            |


## Insights

- The best performing spectrogram features were typically cqtspec, then logspec, and then melspec. However, the best performing features were raw waveforms which was used by RawNet2.
- Pretraining on FoR dataset sped up convergence on the ITW dataset, and the final EER with transfer learning was significantly better than training ITW from scratch.
- ResNet18 is one of the best models for real time deepfake detection, balancing efficient inference time as well as consistent generalization performance.

