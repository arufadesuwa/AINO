import numpy as np
from neural_network import NeuralNetwork

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_train = np.array([[0], [1], [1], [0]])

# X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Y_train = np.array([[1], [0], [0], [1]])

model = NeuralNetwork([2, 3, 1], activation_type='sigmoid')
weighted_sum = 0

epochs = 1000

for epoch in range(epochs):
    total_error = 0
    for i in range(len(X_train)):
        pred = model.forward(X_train[i])
        model.backprop(pred, Y_train[i], n=1.0)

        # print(f"Latihan ke-{i}, Prediksi: {pred}")

print("\n--- Hasil Akhir ---")

err=0

for i in range(len(X_train)):
    pred = model.forward(X_train[i])
    print(f"Input: {X_train[i]}, Target: {Y_train[i]}, Prediksi: {pred}")
    err += (pred - Y_train[i])**2

print(err)

while True:
    x1 = int(input('>'))
    x2 = int(input('>'))

    x = np.array([x1, x2])

    y = model.forward(x)

    if y > 0.5:
        print(1)
    else:
        print(0)
