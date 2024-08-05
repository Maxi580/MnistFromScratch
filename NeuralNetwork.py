import numpy as np
from InputLayer import InputLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer
import ActivationFunctions as AF


def compute_loss(A, Y):
    """measuring how far off your model's predictions are from the true values"""
    m = Y.shape[1]
    epsilon = 1e-15
    loss = -1 / m * np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))
    return loss


class NeuralNetwork:
    def __init__(self, input_shape, hidden_layer_cnt, hidden_layer_neuron_cnt, activation, activation_derivative,
                 output_size):
        self.layers = []

        # Input layer
        self.layers.append(InputLayer(input_shape))

        # Hidden layers
        input_size = input_shape[0]
        for i in range(hidden_layer_cnt):
            self.layers.append(HiddenLayer(input_size, hidden_layer_neuron_cnt, activation, activation_derivative))
            input_size = hidden_layer_neuron_cnt

        # Output layer
        self.layers.append(OutputLayer(hidden_layer_neuron_cnt, output_size, AF.softmax))

    def forward(self, X):
        """Performs forward propagation through the entire network final output is the prediction"""
        A = X
        for layer in self.layers:
            A = layer.forward_propagation(A)
        return A

    def backward(self, Y, learning_rate):
        # Output Layer dA = W[2].T * dZ[2] (If 2 is the output layer)
        dA = self.layers[-1].backward_propagation(Y, learning_rate)

        for layer in reversed(self.layers[1:-1]):
            dA = layer.backward_propagation(dA, learning_rate)

    def train(self, X, Y, epochs, learning_rate, decay_rate):
        for epoch in range(epochs):
            A = self.forward(X)
            loss = compute_loss(A, Y)
            self.backward(Y, learning_rate)

            if epoch % 100 == 0:
                learning_rate *= decay_rate
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        A = self.forward(X)
        return np.argmax(A, axis=0)
