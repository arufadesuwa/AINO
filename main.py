import numpy as np

e = 2.718

class Perceptron:
    def __init__(self, w=None, b=None, activation_type='relu'):
        self.last_z = None
        self.w = w if w is not None else np.random.uniform(-1, 1)
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
            print('cant found function activation, using relu for default')
            return self.relu(y)


biji = Perceptron(activation_type='sigmoid')

print(biji.w, biji.b)

print(biji.sigmoid(2))
print(biji.tanh(2))

print(biji.forward(2))