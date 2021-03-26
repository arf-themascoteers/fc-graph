from abc import ABC, abstractmethod


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons

    @abstractmethod
    def forward(self, inputs, y_true=None):
        pass

    @abstractmethod
    def backward(self, dvalues, y_true=None):
        pass
