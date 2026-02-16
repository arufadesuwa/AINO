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

    @staticmethod
    def sigmoid_derivative(y):
        return y * (1-y)

    @staticmethod
    def tanh_derivative(y):
        return 1 - (y**2)

    @staticmethod
    def relu_derivative(z):
        return 1 if z > 0 else 0

    def get_derivative(self):
        if self.activation_type == 'sigmoid':
            y = self.sigmoid(self.last_z)
            return self.sigmoid_derivative(y)
        elif self.activation_type == 'tanh':
            y = self.tanh(self.last_z)
            return self.tanh_derivative(y)
        elif self.activation_type == 'relu':
            return self.relu_derivative(self.last_z)
        return 1

    def forward(self, x):
        self.last_x = np.array(x)
        y = np.dot(self.w, x)
        y += self.b
        self.last_z = y.copy()
        if self.activation_type == 'sigmoid':
            return self.sigmoid(y)
        elif self.activation_type == 'tanh':
            return self.tanh(y)
        else:
            return self.relu(y)

    def update_weight(self, n, D):

        self.w += n * D *self.last_x
        self.b += n*D
        return D*self.w

if __name__ == '__main__':
    biji = Perceptron(3, activation_type='relu')

    print(biji.w)