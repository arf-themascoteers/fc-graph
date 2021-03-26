from abc import ABC, abstractmethod

class Layer:
    def __init__(self, count_neuron, count_input):
        self.count_neuron = count_neuron
        self.count_input = count_input

    @abstractmethod
    def forward(self, input, y_true=None):
        pass

    @abstractmethod
    def backward(self, dvalues, y_true=None):
        pass


