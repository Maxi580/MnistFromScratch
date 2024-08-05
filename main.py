import numpy as np
from NeuralNetwork import NeuralNetwork
import ActivationFunctions as AF
from tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28 * 28) / 255.0
X_test = X_test.reshape(-1, 28 * 28) / 255.0
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

input_shape = (784,)
hidden_layer_cnt = 3
hidden_layer_neuron_cnt = 256
output_size = 10

nn = NeuralNetwork(input_shape, hidden_layer_cnt, hidden_layer_neuron_cnt,
                   AF.leaky_relu, AF.leaky_relu_derivative, output_size)

# Train the model
epochs = 1000
learning_rate = 0.001
decay_rate = 0.99
nn.train(X_train, y_train, epochs, learning_rate, decay_rate)


# Evaluate the model
def accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

train_predictions = nn.forward(X_train)
test_predictions = nn.forward(X_test)

train_accuracy = accuracy(train_predictions, y_train)
test_accuracy = accuracy(test_predictions, y_test)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
