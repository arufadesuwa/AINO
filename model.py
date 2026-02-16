import numpy as np

from main import Perceptron

class Layer:
    def __init__(self, input_dim, amount=None, activation_type='relu'):
        self.activation_type = activation_type if activation_type is not None else 'relu'
        self.amount = amount if amount is not None else 10
        self.layer = []
        self.input_dim = input_dim
        self.add_perceptron(self.amount)

    def add_perceptron(self, amount):
        for i in range(amount):
            p = Perceptron(dim=self.input_dim, activation_type=self.activation_type)
            self.layer.append(p)

    def forward(self, x):
        y = []
        for i in self.layer:
            y.append(i.forward(x))
        return np.array(y)

if __name__ == '__main__':
    lapisan1 = Layer(3, 10, activation_type='sigmoid')
    lapisan2 = Layer(10, 5, activation_type='sigmoid')

    y = lapisan1.forward([1, 1, 7])
    print(y)
    y = lapisan2.forward(y)
    print(y)

    # print('bobot tiap node')
    # print(lapisan1.layer[3].w)
    #
    # for id, lay in enumerate(lapisan1.layer):
    #     print(lapisan1.layer[id].w)
    #
    # print('---')
    # print('memasukkan input')
    # print(lapisan1.forward([1, 2, 1]))