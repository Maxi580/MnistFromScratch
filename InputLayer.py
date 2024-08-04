import numpy as np


class InputLayer:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, X):
        # Flatten the input if it's a 2D or 3D image
        if len(X.shape) > 2:
            return X.reshape(X.shape[0], -1)
        return X