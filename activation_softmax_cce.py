from activation_softmax import ActivationSoftmax
from loss_cce import LossCategoricalCrossentropy
import numpy as np


class ActivationSoftmaxLossCategoricalCrossEntropy:
    def __init__ ( self ):
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossentropy()
        self.y_true = None

    def forward ( self , inputs ):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, self.y_true)

    def backward ( self , dvalues ):
        samples = len (dvalues)
        if len (self.y_true.shape) == 2 :
            y_true = np.argmax(self.y_true, axis = 1 )
        self.dinputs = dvalues.copy()
        self.dinputs[ range (samples), self.y_true] -= 1
        self.dinputs = self.dinputs / samples

