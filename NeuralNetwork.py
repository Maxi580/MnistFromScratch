import numpy as np
from InputLayer import InputLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer
import ActivationFunctions as AF


class NeuralNetwork:
    def __init__(self, input_shape, hidden_layer_cnt, hidden_layer_neuron_cnt, activation, activation_derivative,
                 output_size):
        self.layers = []

        # Input layer
        self.layers.append(InputLayer(input_shape))

        # Hidden layers
        input_size = np.prod(input_shape)
        for i in range(hidden_layer_cnt):
            self.layers.append(HiddenLayer(input_size, hidden_layer_neuron_cnt, activation, activation_derivative))

        # Output layer
        self.layers.append(OutputLayer(input_size, output_size, AF.softmax))

    def forward(self, X):
        """Performs forward propagation through the entire network final output is the prediction"""
        A = X
        for layer in self.layers:
            A = layer.forward_propagation(A)
        return A

    def backward(self, Y, learning_rate):
        """Performs backpropagation through the entire network, weights and biases are updated"""
        output_layer = self.layers[-1]
        if isinstance(output_layer, OutputLayer):
            dA = output_layer.backward_propagation(Y, self.layers[-2].A, learning_rate)
        else:
            raise ValueError("The last layer should be an OutputLayer")

        # Input Layer is skipped
        for i in range(len(self.layers) - 2, 0, -1):
            current_layer = self.layers[i]
            prev_layer = self.layers[i - 1]
            if isinstance(current_layer, HiddenLayer):
                dA = current_layer.backward_propagation(dA, prev_layer.A, learning_rate)
            else:
                raise ValueError(f"Expected HiddenLayer, got {type(current_layer)}")

    def train(self, X, Y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward propagation
            A = self.forward(X)

            # Compute cost
            cost = self.compute_cost(A, Y)

            # Backward propagation
            self.backward(Y, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost}")

    def compute_cost(self, A, Y):
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        return cost

    def predict(self, X):
        A = self.forward(X)
        return np.argmax(A, axis=0)
