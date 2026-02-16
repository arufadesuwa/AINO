import numpy as np

e = 2.718

class Perceptron:
    def __init__(self, dim, b=None, activation_type='relu'):
        self.last_z = None
        self.w = np.random.uniform(-1, 1, size=dim)
        self.b = b if b is not None else np.random.uniform(-1, 1)
        self.activation_type = activation_type if activation_type is not None else 'relu'

    @staticmethod
    def sigmoid(x):
        y = 1/(1+e**-x)
        return y

    @staticmethod
    def tanh(x):
        y = (e**x-e**-x)/(e**x+e**-x)
        return y

    @staticmethod
    def relu(x):
        y = max(0, x)
        return y

    def forward(self, x):
        y = np.dot(self.w, x)
        y += self.b
        self.last_z = y.copy()
        if self.activation_type == 'sigmoid':
            return self.sigmoid(y)
        elif self.activation_type == 'tanh':
            return self.tanh(y)
        else:
            return self.relu(y)

if __name__ == '__main__':
    biji = Perceptron(3, activation_type='relu')

    print(biji.w)