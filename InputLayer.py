import numpy as np


class InputLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward_propagation(self, X):
        # X Shape: (num_samples, 784)
        return X