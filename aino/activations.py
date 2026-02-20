from .backend import xp

def sigmoid(x: xp.ndarray) -> xp.ndarray:
    """Calculates the Sigmoid activation function."""
    return 1.0 / (1.0 + xp.exp(-x))


def tanh(x: xp.ndarray) -> xp.ndarray:
    """Calculates the Hyperbolic Tangent activation function."""
    return xp.tanh(x)


def relu(x: xp.ndarray) -> xp.ndarray:
    """Calculates the ReLU activation function."""
    return xp.maximum(0.0, x)

def softmax(x: xp.ndarray) -> xp.ndarray:
    """
    Calculates the Softmax activation function.
    Converts raw logits into probabilities that sum to 1.
    """
    shifted_x = x - xp.max(x, axis=1, keepdims=True)
    exp_x = xp.exp(shifted_x)
    return exp_x / xp.sum(exp_x, axis=1, keepdims=True)

def sigmoid_derivative(y: xp.ndarray) -> xp.ndarray:
    """Calculates Sigmoid derivative given the output y."""
    return y * (1.0 - y)


def tanh_derivative(y: xp.ndarray) -> xp.ndarray:
    """Calculates Tanh derivative given the output y."""
    return 1.0 - (y ** 2)


def relu_derivative(z: xp.ndarray) -> xp.ndarray:
    """Calculates ReLU derivative given the logit z."""
    return xp.where(z > 0.0, 1.0, 0.0)


def softmax_derivative(z: xp.ndarray) -> xp.ndarray:
    """
    Calculates Softmax derivative.

    NOTE: In practice, Softmax is almost always paired with Categorical Cross-Entropy loss.
    The mathematical derivative of that pair neatly cancels out to (pred - target).
    Since we compute (pred - target) directly in the network's backprop start,
    this local derivative simply returns an array of 1s to pass the gradient through unchanged.
    """
    return xp.ones_like(z)

def get_derivative(activation_type: str, z: xp.ndarray) -> xp.ndarray:
    """Routes the derivative calculation based on activation type."""
    if activation_type == 'sigmoid':
        y = sigmoid(z)
        return sigmoid_derivative(y)
    elif activation_type == 'tanh':
        y = tanh(z)
        return tanh_derivative(y)
    elif activation_type == 'relu':
        return relu_derivative(z)
    elif activation_type == 'softmax':
        return softmax_derivative(z)

    return xp.ones_like(z)