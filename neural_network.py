from model import Layer

class NeuralNetwork:
    def __init__(self, layer_config, activation_type='relu'):
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

if __name__ == '__main__':
    model = NeuralNetwork([3, 10, 5, 1], activation_type='sigmoid')

    x = [1.5, 1.2, 1.1]
    predict = model.forward(x)

    print(predict)