import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np

from activation_relu import ActivationReLU
from fc import FullyConnected
from layer_dense import LayerDense
from optimizer_adam import OptimizerAdam

nnfs.init()

X, y = spiral_data( samples = 100 , classes = 3 )

fc = FullyConnected(2,3)
fc.add_layer(LayerDense( 2 , 64 ))
fc.add_layer(ActivationReLU(64))
fc.add_layer(LayerDense( 64 , 3 ))
optimizer = OptimizerAdam( learning_rate = 0.05 , decay = 5e-7 )

for epoch in range ( 10001 ):
    fc.forward_backward(X,y)
    optimizer.optimise(fc)

    if not epoch % 100:
        print(f'epoch: {epoch} , ' +
        f'acc: {fc.accuracy:.3f} , ' +
        f'loss: {fc.loss:.3f} , ' +
        f'lr: {optimizer.current_learning_rate:.3f} ' )

plt.scatter(X[:, 0 ], X[:, 1 ], c = y, s = 40 , cmap = 'brg', label = 'initial' )
plt.show()

y = np.argmax(fc.output_layer.output, axis = 1)

plt.scatter(X[:, 0 ], X[:, 1 ], c = y, s = 40 , cmap = 'brg',label = 'predicted', alpha=0.3 )
plt.show()
