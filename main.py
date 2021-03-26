import nnfs
from nnfs.datasets import spiral_data
import numpy as np

from activation_relu import ActivationReLU
from fc import FullyConnected
from output_layer import OutputLayer
from layer_dense import LayerDense
from optimizer_adam import OptimizerAdam

nnfs.init()

X, y = spiral_data( samples = 100 , classes = 3 )

fc = FullyConnected(2,3)
fc.add_layer(LayerDense( 2 , 64 ))
fc.add_layer(ActivationReLU(64))
fc.add_layer(LayerDense( 64 , 3 ))
optimizer = OptimizerAdam( learning_rate = 0.05 , decay = 5e-7 )
fc.print_forward()
for epoch in range ( 10001 ):
    fc.forward(X,y)
    fc.backward(fc.output_layer.output, y)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params_fc(fc)
    optimizer.post_update_params()