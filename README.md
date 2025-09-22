CNN Facial Emotion Detection
This project implements a Convolutional Neural Network (CNN) for facial emotion recognition using the FER2013 dataset. The model is built with PyTorch and classifies grayscale facial images into seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

Features
Custom CNN architecture with batch normalization and dropout for regularization.
Data augmentation (random flips and rotations) for improved generalization.
Class imbalance handling using inverse-frequency class weights in the loss function.
Training, validation, and test splits with per-class accuracy reporting.
Visualization of sample images and confusion matrix.
How to Run
Requirements:

Python 3.x
PyTorch
torchvision
pandas, numpy, matplotlib, seaborn, scikit-learn
Dataset:
Run the script:
Make sure you have a CUDA-enabled GPU.

Improving Accuracy with Inverse Class Weights
The FER2013 dataset is imbalanced—some emotions have many more samples than others. To address this, the code computes inverse-frequency class weights and uses them in the cross-entropy loss function
