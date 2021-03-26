from input_layer import InputLayer
from output_layer import OutputLayer
import numpy as np

class FullyConnected:
    def __init__(self, n_input, n_output):
        self.input_layer = InputLayer(n_input)
        self.output_layer = OutputLayer(n_output, self.input_layer)
        self.input_layer.prev_layer = self.output_layer
        self.accuracy = 0
        self.loss = 0

    def add_layer(self, layer):
        source_layer = self.output_layer.prev_layer

        source_layer.next_layer = layer
        layer.prev_layer = source_layer

        layer.next_layer = self.output_layer
        self.output_layer.prev_layer = layer

    def forward(self, input, output):
        layer = self.input_layer
        while layer is not None:
            input = layer.forward(input, output)
            layer = layer.next_layer

        predictions = np.argmax(self.output_layer.output, axis=1)
        self.accuracy = np.mean(predictions == output)
        self.loss = self.output_layer.calculated_loss
        return self.output_layer.output

    def backward(self, dvalues, y_true=None):
        layer = self.output_layer
        while layer != self.input_layer:
            layer.backward(dvalues, y_true)
            dvalues = layer.dinputs
            layer = layer.prev_layer

    def forward_backward(self, input, output):
        dvalues = self.forward(input, output)
        self.backward(dvalues, output)

    def print_forward(self):
        layer = self.input_layer
        print("Machine")
        print("=======")
        while layer is not None:
            print(f"{layer.n_neurons} - {type(layer).__name__}")
            layer = layer.next_layer