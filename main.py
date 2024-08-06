import numpy as np
from NeuralNetwork import NeuralNetwork
import ActivationFunctions as AF
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28 * 28) / 255.0
X_test = X_test.reshape(-1, 28 * 28) / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# one-dimensional array with 784 elements, each value corresponds to a pixel
input_shape = (784,)
hidden_layer_cnt = 3
hidden_layer_neuron_cnt = 256
output_size = 10

nn = NeuralNetwork(input_shape, hidden_layer_cnt, hidden_layer_neuron_cnt,
                   AF.leaky_relu, AF.leaky_relu_derivative, output_size)

# Train the model
epochs = 1000
learning_rate = 0.001
drop_factor = 0.95
batch_size = 2048  # There are 60.000 Images in The Dataset, on each Epoch we use batch_size amount
epoch_drop = 500
nn.train(X_train, y_train, epochs, learning_rate, drop_factor, epoch_drop, batch_size)


# Evaluate the model
def accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))


train_predictions = nn.forward(X_train)
test_predictions = nn.forward(X_test)

train_accuracy = accuracy(train_predictions, y_train)
test_accuracy = accuracy(test_predictions, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
