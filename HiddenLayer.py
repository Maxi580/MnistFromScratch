import numpy as np


class HiddenLayer:
    def __init__(self, input_size, output_size, activation_function, activation_function_derivative):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.activation = activation_function
        self.activation_derivative = activation_function_derivative
        self.b = np.random.randn(output_size, 1) * 0.01  # Weights initialized to small numbers
        self.Z = None  # The Output before Activation, Gets Saved as we need it for backwards propagation
        self.A_prev = None  # The Input of the previous Layer, needed for backpropagation (Weight Calculation)

    def forward_propagation(self, A_prev):
        self.A_prev = A_prev  # (num_samples, num_features_in)
        self.Z = np.dot(self.W, A_prev.T) + self.b  # Self.W = (num_neurons_out, num_features_in)
        # Result:  (num_neurons_out, num_samples)
        A = self.activation(self.Z)  # (num_neurons_out, num_samples)
        return A.T  # Gets Transposed to get (num_samples, num_features)

    def backward_propagation(self, dA, learning_rate):
        m = self.A_prev.shape[0]  # A_prev =  (num_samples, num_features) => batch_size = m = num_samples

        dZ = dA.T * self.activation_derivative(self.Z)

        dW = 1 / m * np.dot(dZ, self.A_prev)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return np.dot(self.W.T, dZ).T
