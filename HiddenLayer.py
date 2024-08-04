import numpy as np


class HiddenLayer:
    def __init__(self, input_size, output_size, activation_function, activation_function_derivative):
        self.W = np.random.randn(output_size, input_size) * 0.01
        self.activation = activation_function
        self.activation_derivative = activation_function_derivative
        self.b = np.zeros((output_size, 1))
        self.Z = None
        self.A = None

    def forward_propagation(self, A_prev):
        self.Z = np.dot(self.W, A_prev) + self.b
        self.A = self.activation(self.Z)
        return self.A

    def backward_propagation(self, dA, A_prev, learning_rate):
        """dA here represents W.T * dZ from the Layer that comes afterward´s """
        m = A_prev.shape[1]

        dZ = dA * self.activation_derivative(self.Z)

        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dA_prev
