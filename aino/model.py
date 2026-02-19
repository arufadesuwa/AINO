import numpy as np
from typing import List, Optional, Any
from .layer import Layer


class NeuralNetwork:
    def __init__(self, layer_config: List[int], activation_type: str = 'relu'):
        self.config: List[int] = layer_config
        self.activation_type: str = activation_type if activation_type is not None else 'relu'
        self.layers: List[Layer] = []

        for i in range(len(layer_config) - 1):
            i_dim = layer_config[i]
            neuron_count = layer_config[i + 1]

            new_layer = Layer(i_dim, neuron_count, activation_type=self.activation_type)
            self.layers.append(new_layer)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        current_input = x
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def _backprop(self, pred: np.ndarray, target: np.ndarray, n: float = 0.1) -> None:
        error = pred - target
        for layer in reversed(self.layers):
            error = layer.backward(error)
            layer.update(n)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        return self._forward(X)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, n: float = 0.1, batch_size: int = 32,
            verbose: bool = False) -> None:
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        n_samples = len(X)
        n_outputs = y.shape[1]

        print(f"Start training AINO model (Mini-Batch) over {epochs} epochs...")
        print(f"Batch Size: {batch_size} | Samples: {n_samples}")

        # 2. Siapkan array indeks untuk shuffling yang efisien
        indices = np.arange(n_samples)

        for epoch in range(epochs):
            np.random.shuffle(indices)

            total_error = 0.0

            for i in range(0, n_samples, batch_size):
                idx_batch = indices[i: i + batch_size]
                X_batch = X[idx_batch]
                y_batch = y[idx_batch]

                pred = self._forward(X_batch)

                total_error += np.sum((y_batch - pred) ** 2)

                self._backprop(pred, y_batch, n)

            if verbose and (epoch % (max(1, epochs // 10)) == 0):
                avg_loss = total_error / (n_samples * n_outputs)
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

        print("Training is done!")

    def save(self, filename: str = "model.dit") -> None:
        if not filename.endswith(".dit"):
            filename += ".dit"

        data = {
            'config': np.array(self.config),
            'activation': np.array(self.activation_type)
        }

        for i, layer in enumerate(self.layers):
            data[f"layer_{i}_w"] = layer.w
            data[f"layer_{i}_b"] = layer.b

        with open(filename, 'wb') as f:
            np.savez(f, **data)

        print(f"Successfully saved model to '{filename}'")

    @staticmethod
    def load(filename: str) -> Optional['NeuralNetwork']:
        if not filename.endswith(".dit"):
            print(f"Warning: '{filename}' isn't a .dit model...")

        try:
            with open(filename, 'rb') as f:
                data = np.load(f, allow_pickle=True)
                config = data['config'].tolist()
                activation = str(data['activation'])

                print(f"Architecture: {config}, Non-linearity: {activation}")
                model_baru = NeuralNetwork(config, activation_type=activation)

                for i, layer in enumerate(model_baru.layers):
                    layer.w = data[f"layer_{i}_w"]
                    layer.b = data[f"layer_{i}_b"]

                print("AINO model ready")
                return model_baru

        except Exception as e:
            print(f"Error when trying to load model: {e}")
            return None