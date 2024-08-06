import numpy as np

class OutputLayer:
    def __init__(self, input_size, output_size, activation_function):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.activation = activation_function
        self.b = np.random.randn(output_size, 1) * 0.01
        self.A = None # The Output after Activation, Gets Saved as we need it for backwards propagation
        self.A_prev = None  # The Input of the previous Layer, needed for backpropagation (Weight Calculation)

    def forward_propagation(self, A_prev):
        self.A_prev = A_prev  # (num_samples, num_features_in)
        Z = np.dot(self.W, A_prev.T) + self.b  # Self.W = (num_neurons_out, num_features_in)
        # Result:  (num_neurons_out, num_samples)
        self.A = self.activation(Z)  # (num_neurons_out, num_samples)
        return self.A.T  # Gets Transposed to get (num_samples, num_features)

    def backward_propagation(self, Y, learning_rate):
        # Y is the correct output: Y[1] = [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        m = len(Y)  # len(Y) is the amount of samples in the provided data and therefore batch_size

        dZ = self.A - Y.T

        dW = 1 / m * np.dot(dZ, self.A_prev)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return np.dot(self.W.T, dZ).T
