This project implements a Feedforward Neural Network (FNN) from scratch in PyTorch to classify handwritten digits from the MNIST dataset.
We use custom training, validation, and testing loops, with a focus on understanding core machine learning concepts without relying on high-level APIs.

Table of Contents
Project Overview

Model Architecture

Dataset

Installation

Running the Project

Experiments

Results

Future Work

License

Project Overview
The goal is to classify handwritten digits (0-9) using a simple feedforward neural network trained with stochastic gradient descent.
The dataset is split into:

60% training

20% validation

20% testing

The model is evaluated based on validation and test accuracy.

Model Architecture
Input Layer: 784 neurons (28x28 flattened images)

Hidden Layer 1: 128 neurons, ReLU activation

Hidden Layer 2: 64 neurons, ReLU activation

Output Layer: 10 neurons (one for each digit class)

Optimizer: Stochastic Gradient Descent (SGD)
Loss Function: Cross-Entropy Loss
Learning Rate: 0.01
Epochs: 10

Dataset
We use the MNIST dataset, a large collection of handwritten digits commonly used for training various image processing systems.

Training samples: 60%

Validation samples: 20%

Test samples: 20%

Splitting is done using stratified sampling to ensure balanced class distribution.

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/handwritten-digit-recognition-fnn.git
cd handwritten-digit-recognition-fnn
Install required packages:

bash
Copy
Edit
pip install torch torchvision scikit-learn matplotlib
Running the Project
Open the Jupyter Notebook:

bash
Copy
Edit
jupyter notebook ML_PROJECT.ipynb
Run all the cells step by step to:

Download and prepare the dataset

Define the model

Train the model

Evaluate the model on validation and test sets

Experiments
In the notebook, you can easily modify:

Number of hidden layers or neurons (to see effects on accuracy)

Batch size

Learning rate

Number of epochs

Optimizer type (try Adam instead of SGD)

Results
Achieved high accuracy on the MNIST validation and test sets.

Observed effective training with a relatively simple architecture and basic hyperparameters.

You can add detailed accuracy/loss plots here if available.

Future Work
Add CNN (Convolutional Neural Network) models for better accuracy

Experiment with different optimizers (Adam, RMSprop)

Perform hyperparameter tuning (learning rates, batch sizes)

Add early stopping to prevent overfitting

License
This project is licensed under the MIT License. See the LICENSE file for details.
