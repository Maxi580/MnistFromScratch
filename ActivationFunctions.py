import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)


def softmax_derivative(x):
    # Note: This is a simplification. The true Jacobian of softmax is more complex.
    # This simplification works when used with cross-entropy loss.
    s = softmax(x)
    return s * (1 - s)


def leaky_relu(x, a=0.01):
    return np.where(x > 0, x, a * x)


def leaky_relu_derivative(x, a=0.01):
    return np.where(x > 0, 1, a)
