import numpy as np


class ActivationFunctions:
    def activation(self, z, activation_function):
        if activation_function == "SIGMOID":
            return 1 / (1 + np.exp(-z))
        if activation_function == "TANH":
            return np.tanh(z)
        if activation_function == "RELU":
            return np.maximum(0, z)

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def der_activation(self, z, activation_function):
        if activation_function == "SIGMOID":
            return z * (1 - z)
        if activation_function == "TANH":
            return 1 - z ** 2
        if activation_function == "RELU":
            return np.where(z > 0, 1, 0)


act = ActivationFunctions()
