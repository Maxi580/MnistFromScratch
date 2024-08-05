import numpy as np


def clip_gradients(grad, clip_value=1):
    """exploding gradients problem"""
    return np.clip(grad, -clip_value, clip_value)


class OutputLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.activation = activation_function
        self.b = np.zeros((output_size, 1))
        self.Z = None
        self.A = None
        self.A_prev = None  # Store the input to this layer

    def forward_propagation(self, A_prev):
        self.A_prev = A_prev  # Store for use in backpropagation
        self.Z = np.dot(self.W, A_prev.T) + self.b
        self.A = self.activation(self.Z)
        return self.A.T

    def backward_propagation(self, Y, learning_rate):
        m = Y.shape[0]

        dZ = self.A - Y.T

        dW = 1 / m * np.dot(dZ, self.A_prev)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return np.dot(self.W.T, dZ).T
