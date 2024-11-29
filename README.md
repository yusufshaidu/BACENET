# ML-Potentials (Under development)



## Getting started

Implementing the modified Behler Parrinello symmentry functions with learnable species embedding, gaussian spreads and the three body resolution in the descriptors.
The species embedding starts with a one-hot encoding of species and pass it through a single layer multilayer perceptron of size 16 and returns a single number per species.
The weights are learn from data.

## todolist
- Implement input normalization
- start from known species embedding rather than a one-hot-encoding [https://github.com/WMD-group/ElementEmbeddings] 



