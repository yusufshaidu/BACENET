import flax.nnx as nnx
from typing import Any, Sequence

class SimpleNN(nnx.Module):    
    def __init__(self,  n_features: int, layer_sizes: Sequence[int], activations: Sequence[str],
                 *, rngs: nnx.Rngs):
    

        self.n_features = n_features
        self._activations = activations
        self._layers = []
        for i, l in enumerate(layer_sizes[:-1]):
            if i == 0:
                self._layers.append(nnx.Linear(n_features, layer_sizes[i], rngs=rngs))
            else:
                self._layers.append(nnx.Linear(layer_sizes[i], layer_sizes[i+1], rngs=rngs))
                
            print(i,layer_sizes[i], layer_sizes[i+1])
    def acts(self, activation):
        ACTIVATIONS = {
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "tanh": jax.nn.tanh,
        "selu": nnx.selu,
        "sigmoid": jax.nn.sigmoid,
        'silu': jax.nn.silu,
        'linear': jax.nn.identity
        } 
        return ACTIVATIONS[activation]
    #@nn.compact
    def __call__(self, x):
        x = x.reshape(x.shape[0], self.n_features)
        for i, layer in enumerate(self._layers):
            x = self.acts(self._activations[i])(layer(x))
        return x
