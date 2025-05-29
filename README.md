## ML-Potentials using learnable species embedding and bessels radial functions (Under development)

Implementing the modified Behler Parrinello symmentry functions with learnable species embedding, radial bessels functions [https://arxiv.org/abs/2003.03123] and the three body angular centers of the descriptors.
The species embedding starts with a one-hot encoding of species and pass it through a single layer perceptron of size 64 and returns species encoder with user specified dimension.
All weights are learn from data. Neighborlist are computed with ASE and stored alongside the positions, atomic numbers, C6, nats, cells, first neighbor index(=i), second neighbor inde (=j), the shift vectors (=S), energy and forces.

## Equations
 # Redial basis functions
 The radial functions employed in this code are given by:
 $$R_n(r) = \sqrt{\frac{2}{r_c}} \frac{{\rm sin}(\frac{n\pi}{r_c}k_n r)}{r} f_c(r,r_c)$$
 To encode species information rather than creating NNP for each species channels, we start from one-hot encoder which creates a long vector of zeros with only the possible corresponding to the atomic number of species takes a value of 1. This one-hot encoder is then pass through a fully trainable user-defined feedforward NNs that returns a fixed vector $S_i$ that represent the species $i$. The species information is then multiplied by the radial function in two possible ways: a tensor product $S(s_1,s_2)=S_{s_1} \otimes S_{s_2}$ or a dot product $S(s_1,s_2)=S_{s_2} . S_{s_1}$.

The species resolved radial functions are then give by:
 $$R_n(r,s_1,s_2) = \sqrt{\frac{2}{r_c}} \frac{{\rm sin}(\frac{n\pi}{r_c}k_n r)}{r} f_c(r,r_c) S(s_1,s_2) $$ 

## todolist
- (priority 1) implement ASE interface
- (priority) implement van der Waals and long range electrostatics!!
- implement a lammps interface for production runs
- Implement input normalization
- start from known species embedding rather than a one-hot-encoding [https://github.com/WMD-group/ElementEmbeddings] 



