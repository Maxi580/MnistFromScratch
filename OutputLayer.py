import numpy as np


class OutputLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.activation = activation_function
        self.b = np.zeros((output_size, 1))
        self.Z = None
        self.A = None

    def forward_propagation(self, A_prev):
        self.Z = np.dot(self.W, A_prev) + self.b
        self.A = self.activation(self.Z)
        return self.A

    def backward_propagation(self, Y, A_prev, learning_rate):
        m = A_prev.shape[1]

        dZ = self.A - Y

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dA_prev
