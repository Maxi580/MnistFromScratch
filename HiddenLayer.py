import numpy as np


def clip_gradients(grad, clip_value=1):
    return np.clip(grad, -clip_value, clip_value)


class HiddenLayer:
    def __init__(self, input_size, output_size, activation_function, activation_function_derivative):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.activation = activation_function
        self.activation_derivative = activation_function_derivative
        self.b = np.zeros((output_size, 1))
        self.Z = None
        self.A = None

    def forward_propagation(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(self.W, A_prev.T) + self.b
        self.A = self.activation(self.Z)
        return self.A.T

    def backward_propagation(self, dA, learning_rate):
        m = self.A_prev.shape[0]

        dZ = dA.T * self.activation_derivative(self.Z)

        dW = 1 / m * np.dot(dZ, self.A_prev)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return np.dot(self.W.T, dZ).T