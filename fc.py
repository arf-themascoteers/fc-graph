from input_layer import InputLayer
from output_layer import OutputLayer
import numpy as np

class FullyConnected:
    def __init__(self, n_input, n_output):
        self.input_layer = InputLayer(n_input)
        self.output_layer = OutputLayer(n_output, self.input_layer)
        self.input_layer.prev_layer = self.output_layer

    def add_layer(self, layer):
        source_layer = self.output_layer.prev_layer

        source_layer.next_layer = layer
        layer.prev_layer = source_layer

        layer.next_layer = self.output_layer
        self.output_layer.prev_layer = layer

    def forward(self, input, output):
        layer = self.input_layer
        while layer is not None:
            layer.forward(input, output)
            input = layer.output
            layer = layer.next_layer

        predictions = np.argmax(self.output_layer.output, axis=1)

        accuracy = np.mean(predictions == output)
        print(accuracy)

    def backward(self, dvalues, y_true=None):
        layer = self.output_layer
        while layer != self.input_layer:
            layer.backward(dvalues, y_true)
            dvalues = layer.dinputs
            layer = layer.prev_layer

    def print_forward(self):
        layer = self.input_layer
        while layer is not None:
            print(f"^{layer.n_neurons}")
            layer = layer.next_layer