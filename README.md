## ML-Potentials: Exploring the equivalence of Behler-Parrinello symmetry functions and atomic cluster expansion, an alternative implementation with linear scaling in number of neighbors, learnable species embedding and higher body order terms

Implementing the modified Behler Parrinello symmentry functions with learnable species embedding, radial bessels functions [https://arxiv.org/abs/2003.03123] and the three body angular centers of the descriptors.
The species embedding starts with a one-hot encoding of species and pass it through a single layer perceptron of size 64 and returns species encoder with user specified dimension.
All weights are learn from data. Neighborlist are computed with ASE and stored alongside the positions, atomic numbers, C6, nats, cells, first neighbor index(=i), second neighbor inde (=j), the shift vectors (=S), energy and forces.

## Equations
 # Redial basis functions
 The radial functions employed in this code are given by:
 
 $$R_n(r) = \sqrt{\frac{2}{r_c}} \frac{{\rm sin}(\frac{n\pi}{r_c}k_n r)}{r} f_c(r,r_c)$$
 
 To encode species information rather than creating NNP for each species channels, we start from one-hot encoder which creates a long vector of zeros with only the possible corresponding to the atomic number of species takes a value of 1. This one-hot encoder is then pass through a fully trainable user-defined feedforward NNs that returns a fixed vector $S_i$ that represent the species $i$. The species information is then multiplied by the radial function in two possible ways: a tensor product $S(s_1,s_2)=S_{s_1} \otimes S_{s_2}$ or a dot product $S(s_1,s_2)=S_{s_2} . S_{s_1}$.

The species resolved radial functions are then give by:

 $$R_n(r,s_1,s_2) = \sqrt{\frac{2}{r_c}} \frac{{\rm sin}(\frac{n\pi}{r_c}k_n r)}{r} f_c(r,r_c) S(s_1,s_2)$$ 

 # Angular descriptors
 We start with the Behler-Parrinello angular descriptors given by
 
 $$G_i^{(3)}(s) = 2^{1-\zeta}\sum_{jk} (1+\lambda_s {\rm cos}(\theta_{ijk}))^\zeta R_{ijs} R_{iks}$$

where $R_{ijs} = R_n(r_{ij},s_1,s_2) fc(r_{ij, r_c})$
## todolist
- implement a lammps interface for production runs

