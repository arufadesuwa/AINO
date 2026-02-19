"""
AINO (Artificial Intelligence Neural Objects)
=============================================

AINO is a lightweight, educational Neural Network library built from scratch using NumPy.
It is designed to help users understand the fundamental concepts of Deep Learning,
such as Forward Pass, Backpropagation, and highly optimized Vectorized Matrix Operations.

Key Features:
-------------
- **NeuralNetwork**: Create, train, and save/load Multi-Layer Perceptron (MLP) models.
- **Layer**: Manages vectorized operations (weights and biases matrices) for high-speed computation.

Usage Example:
--------------
>>> import aino
>>> import numpy as np
>>> # Create a model: 2 inputs, 4 hidden neurons, 1 output (XOR problem)
>>> model = aino.NeuralNetwork([2, 4, 1], activation_type='sigmoid')
>>> X = np.array([[0,0], [0,1], [1,0], [1,1]])
>>> y = np.array([[0], [1], [1], [0]])
>>> model.fit(X, y, epochs=1000, n=0.1)

Author: Arufa
License: MIT
"""

from .model import NeuralNetwork
from .layer import Layer

__all__ = ["NeuralNetwork", "Layer"]

__version__ = "1.0.0"
__author__ = "Arufa"