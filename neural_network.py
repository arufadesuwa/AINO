from model import Layer
import numpy as np


class NeuralNetwork:
    def __init__(self, layer_config, activation_type='relu'):
        self.config = layer_config
        self.activation_type = activation_type if activation_type is not None else 'relu'
        self.layer = []

        for i in range(len(layer_config) - 1):
            i_dim = layer_config[i]
            neuron = layer_config[i + 1]

            new_layer = Layer(i_dim, neuron, activation_type=self.activation_type)
            self.layer.append(new_layer)

    def forward(self, x):
        current_input = x
        for layer in self.layer:
            current_input = layer.forward(current_input)
        return current_input

    def backprop(self, pred, target, n=0.1):
        error = target - pred
        for layer in reversed(self.layer):
            error = layer.backward(error, n)

    def save(self, filename="model.dit"):
        if not filename.endswith(".dit"):
            filename += ".dit"

        data = {}

        data['config'] = np.array(self.config)
        data['activation'] = np.array(self.activation_type)

        for i, layer in enumerate(self.layer):
            w_layer = np.array([p.w for p in layer.layer])
            b_layer = np.array([p.b for p in layer.layer])
            data[f"layer_{i}_w"] = w_layer
            data[f"layer_{i}_b"] = b_layer

        with open(filename, 'wb') as f:
            np.savez(f, **data)

        print(f"Berhasil! Jiwa model telah disegel dalam '{filename}' 🔮")

    @staticmethod
    def load(filename):
        if not filename.endswith(".dit"):
            print(f"Peringatan: '{filename}' sepertinya bukan format .dit asli...")

        try:
            with open(filename, 'rb') as f:
                data = np.load(f, allow_pickle=True)

                config = data['config'].tolist()
                activation = str(data['activation'])

                print(f"architerture: {config}, non-linearity: {activation}")

                model_baru = NeuralNetwork(config, activation_type=activation)

                for i, layer in enumerate(model_baru.layer):
                    w_layer = data[f"layer_{i}_w"]
                    b_layer = data[f"layer_{i}_b"]
                    for j, perceptron in enumerate(layer.layer):
                        perceptron.w = w_layer[j]
                        perceptron.b = b_layer[j]

                print("AINO model ready")
                return model_baru

        except FileNotFoundError:
            print(f"Gawat! File '{filename}' tidak ditemukan.")
            return None
        except Exception as e:
            print(f"File .dit rusak atau korup: {e}")
            return None


if __name__ == '__main__':
    model = NeuralNetwork([3, 10, 5, 2], activation_type='sigmoid')

    x = [1.5, 1.2, 1.1]
    predict = model.forward(x)
    print(f"Prediksi Awal: {predict}")

    model.save("percobaan_pertama")

    print("\n--- Memuat Ulang ---")
    model_reinkarnasi = NeuralNetwork.load("percobaan_pertama.dit")

    if model_reinkarnasi:
        predict_baru = model_reinkarnasi.forward(x)
        print(f"Prediksi Setelah Load: {predict_baru}")