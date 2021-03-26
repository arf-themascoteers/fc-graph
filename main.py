import nnfs
from nnfs.datasets import spiral_data
import numpy as np

from activation_relu import ActivationReLU
from activation_softmax_cce import ActivationSoftmaxLossCategoricalCrossEntropy
from layer_dense import LayerDense
from optimizer_adam import OptimizerAdam

nnfs.init()

X, y = spiral_data( samples = 100 , classes = 3 )
dense1 = LayerDense( 2 , 64 )
activation1 = ActivationReLU(64)
dense2 = LayerDense( 64 , 3 )

loss_activation = ActivationSoftmaxLossCategoricalCrossEntropy(3,1)
optimizer = OptimizerAdam( learning_rate = 0.05 , decay = 5e-7 )
for epoch in range ( 10001 ):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output,y)
    predictions = np.argmax(loss_activation.output, axis = 1 )
    if len (y.shape) == 2 :
        y = np.argmax(y, axis = 1 )
    accuracy = np.mean(predictions == y)
    if not epoch % 100 :
        print ( f'epoch: {epoch} , ' +
        f'acc: {accuracy:.3f} , ' +
        f'loss: {loss:.3f} , ' +
        f'lr: {optimizer.current_learning_rate} ' )

    # Backward pass
    loss_activation.backward(loss_activation.output,y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()