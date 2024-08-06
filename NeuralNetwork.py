import numpy as np
from InputLayer import InputLayer
from HiddenLayer import HiddenLayer
from OutputLayer import OutputLayer
import ActivationFunctions as AF


def step_decay(initial_lr, drop_factor, epochs_drop, epoch):
    return initial_lr * (drop_factor ** (epoch // epochs_drop))


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
        input_size = input_shape[0]  # The First Layer gets 784 input from the input Layer
        for i in range(hidden_layer_cnt):
            self.layers.append(HiddenLayer(input_size, hidden_layer_neuron_cnt, activation, activation_derivative))
            input_size = hidden_layer_neuron_cnt
            # However the next Hidden Layer gets as much input as there were neurons in the Layer before

        # Output layer
        self.layers.append(OutputLayer(hidden_layer_neuron_cnt, output_size, AF.softmax))

    def forward(self, X):
        """Performs forward propagation through the entire network final output is the prediction
           Gets a image vector, passes it through each layer and gets a prediction back, straight forward"""
        A = X  # X Shape: (num_samples, 784)
        for layer in self.layers:
            A = layer.forward_propagation(A)
        return A

    def backward(self, Y, learning_rate):
        """Starts at the end and does backwards propagation. Output Layer is handled differently as it has different
        input because the Error in backwards-propagation gets calculated differently from hidden layers. Since this
        is the only change it should probably be improved sometime. Also Input Layer gets skipped as there are no
        weights or biases to be improved etc. """
        # Output Layer dA = W[2].T * dZ[2] (If 2 is the output layer)
        # This is the Part you need to calculate the  error in the previous part which is why it gets passed
        dA = self.layers[-1].backward_propagation(Y, learning_rate)

        for layer in reversed(self.layers[1:-1]):
            dA = layer.backward_propagation(dA, learning_rate)

    def train(self, X, Y, epochs, learning_rate, drop_factor, epoch_drop, batch_size):
        """Well, handles Batches of Data. Therefore X can be imagined as (num_samples, 784).
            Y is the correct Result which is needed for loss and backwards propagation, A is the prediction"""

        initial_lr = learning_rate
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]

            # Creates random batches of batch_size so not all Data gets used in one epoch
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i + batch_size]  # Shape: (num_samples, 784)
                Y_batch = Y_shuffled[i:i + batch_size]  # Shape: (num_samples, 784)

                A = self.forward(X_batch)
                loss = compute_loss(A, Y_batch)
                self.backward(Y_batch, learning_rate)

            if epoch % 100 == 0:
                learning_rate = step_decay(initial_lr, drop_factor, epoch_drop, epoch)
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        """axis=0 refers to the first axis (rows in a 2D array)
           axis=1 refers to the second axis (columns in a 2D array)
           means we're looking for the maximum value in each column
           Each column represents the predictions for one input sample"""
        A = self.forward(X)
        return np.argmax(A, axis=0)
