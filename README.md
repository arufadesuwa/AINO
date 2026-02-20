# AINO (Aino is Neural Operation)
![AINO Banner](https://raw.githubusercontent.com/arufadesuwa/AINO/f933ba6c0186b904ef871e4b118621878d97a07a/assets/aino.webp)

> **"Aino is Neural Operation."** > A custom-built, highly optimized Deep Learning framework built from scratch using pure Python, and NumPy.

[![PyPI version](https://badge.fury.io/py/aino.svg)](https://badge.fury.io/py/aino)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌟 Inspiration

This project was born out of curiosity after watching this inspiring video:  
[**MIT Introduction to Deep Learning | 6.S191**](https://youtu.be/alfdI7S6wCY?si=MPqH4F2EiP3U67t-)

I didn't want to just `import tensorflow` and call it a day. I wanted to see **directly** how the magic happens under the hood. I wanted to feel the weight of the matrices, understand the flow of the gradients, and build the brain from scratch.

## ⚡ The Tech Stack (Hardware Agnostic)

AINO is built to be educational yet blazingly fast. It uses an **Agnostic Backend**:
* **CPU Mode:** Uses pure `NumPy` with contiguous memory optimization.
* **GPU Mode:** Automatically detects and switches to `CuPy` if an NVIDIA GPU is available, providing massive parallel acceleration without changing a single line of your code.
* **No Black Boxes:** Every Forward Pass and Backpropagation step is manually calculated.

## ✨ Features

* **Flexible Architecture:** Define any number of layers and neurons (e.g., `[784, 128, 64, 10]`).
* **Vectorized Operations:** Dropped slow loop-based perceptrons in favor of highly optimized matrix multiplications.
* **Mini-Batch Gradient Descent:** Train on large datasets (like MNIST) efficiently.
* **Activation Functions:** Supports `Sigmoid`, `ReLU`, and `Tanh`.
* **Universal Serialization:** Safely save (`.dit`) and load models across different machines, whether they have a GPU or not.

## 🧠 What I Learned

Building AINO from the ground up gave me insights that high-level libraries often hide:

1. **From OOP to Vectorization:** I initially built the network iterating over individual Perceptrons. I quickly learned that Python loops are slow. Refactoring the `Layer` class to use pure Matrix Calculus (`np.dot`) reduced training time from 32 minutes to just 19 seconds!
2. **The Calculus of Backpropagation:**
   I implemented the **Chain Rule** manually, computing derivatives for activations and understanding how error gradients propagate from the output back to the input layers.
3. **Memory Management & Hardware:**
   I learned the critical difference between RAM and VRAM, how to use `ascontiguousarray` for CPU caching, and how to safely bridge data between CPU and GPU using `CuPy`.

## 💻 Usage Example

```python
from aino.model import NeuralNetwork

# Create a network for MNIST (784 inputs, 2 hidden layers, 10 outputs)
model = NeuralNetwork([784, 128, 64, 10], activation_type='tanh')

# Train using Mini-Batch Gradient Descent (Auto CPU/GPU)
model.fit(X_train, y_train, epochs=100, n=0.01, batch_size=32, verbose=True)

# Make predictions
predictions = model.predict(X_test)

# Save the universally loadable .dit model
model.save('aino_mnist.dit')
```

Built with ❤️ using Python.