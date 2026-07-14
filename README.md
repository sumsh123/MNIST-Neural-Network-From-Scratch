# NeuroCanvas
## Handwritten Digit Recognition Neural Network Built From Scratch

A handwritten digit recognition system built completely from scratch using Python and NumPy.

This project focuses on understanding the foundations of deep learning by implementing a neural network without relying on high-level frameworks such as TensorFlow or PyTorch.

The model learns to recognize handwritten digits from the MNIST dataset using manually implemented forward propagation, backpropagation, gradient descent, activation functions, and loss calculation.

The project includes an interactive drawing interface where users can draw a digit and receive predictions with confidence scores.

---

## Features

- Neural network implemented from scratch using NumPy
- Dense fully connected layers
- ReLU activation function
- Softmax output layer
- Cross entropy loss
- Forward propagation
- Backpropagation
- Gradient descent optimization
- MNIST dataset training
- Interactive handwritten digit recognition
- Prediction confidence output
- Neural network visualization (in development)

---

## Neural Network Architecture

```
Input Layer
784 neurons (28×28 image pixels)

        ↓

Dense Layer
128 neurons

        ↓

ReLU Activation

        ↓

Dense Layer
64 neurons

        ↓

ReLU Activation

        ↓

Output Layer
10 neurons

        ↓

Softmax
10 probability outputs
```

---

## Results

Dataset:

MNIST Handwritten Digits Dataset

Accuracy:

~96%

The model was trained without TensorFlow or PyTorch to understand the mathematics and internal mechanisms behind neural networks.

---

## Technologies Used

Python  
NumPy  
Pygame  
Pillow  
Matplotlib  
Scikit-learn  

---

## How It Works

### Forward Propagation

The input image is passed through multiple layers:

```
Image Pixels
      ↓
Dense Layers
      ↓
Activation Functions
      ↓
Output Probabilities
      ↓
Prediction
```

### Loss Calculation

The predicted probabilities are compared with the actual label using cross entropy loss.

### Backpropagation

The network calculates gradients and updates weights to reduce prediction error.

Training process:

```
Forward Pass
      ↓
Loss Calculation
      ↓
Backward Pass
      ↓
Weight Updates
```

---

## Project Structure

```
NeuroCanvas/

├── main.py
├── train.py
├── network.py
├── layers.py
├── activations.py
├── loss.py
├── mnist_loader.py
├── model.py
├── draw.py
└── model.pkl
```

---

## Learning Journey

This project was built while studying deep learning fundamentals through DeepLearning.AI's introductory deep learning course.

The goal was to move beyond simply using AI libraries and understand how neural networks work internally by implementing the core mathematics and algorithms myself.

Through this project, I explored:

- Neural network architecture
- Matrix operations
- Activation functions
- Optimization algorithms
- Gradient-based learning
- Model evaluation

---

## Future Improvements

- Real-time neural network visualization
- Convolutional Neural Network implementation
- Better handwriting preprocessing
- Web deployment
- Larger datasets
- GPU acceleration

---
```
