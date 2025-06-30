## (Intro) BACE-Net (Behler–Parrinello Atomic Cluster Expansion neural Networks):
 -BACE-Net is a symmetry-function-based interatomic potential that combines the body-order and linear-scaling benefits of ACE with the expressive power of neural networks and learnable species embeddings.

Implements Behler Parrinello symmentry functions with learnable species embedding, linear scaling like ACE with higher body-order descriptors, and radial bessels functions [https://arxiv.org/abs/2003.03123]
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
 
 $$G_i^{(3)}(s) = 2^{1-\zeta}\sum_{jk} (\frac{1+\lambda_s {\rm cos}(\theta_{ijk})}{2})^\zeta R_{ijs} R_{iks}$$

where $R_{ijs} = R_n(r_{ij},s_1,s_2) fc(r_{ij, r_c})$

Expanding over the integer $\zeta$ and the $cos(\theta)$, gives:

$$G_{\bf s}^3 = \frac{2}{2^{\zeta}}\sum_{l=0}^{\zeta}\lambda^l\frac{\zeta!}{l!(\zeta-l)!} \sum_{l_xl_yl_z} \frac{l!}{l_x!l_y!l_z!} (G_{i,l_xl_yl_z,{\bf s}})^2$$
$$G_{\bf s}^3= \frac{2}{2^{\zeta}}\sum_{l=0}^{\zeta}\lambda^l G_{{\bf s}l}^3$$
where
$$G_{{\bf s}l}^3 =\sum_{l_xl_yl_z} \frac{\zeta!}{l!(\zeta-l)!} \frac{l!}{l_x!l_y!l_z!} (\mathcal{G}_{i,l_xl_yl_z,{\bf s}})^2,
$$
and 
$$\mathcal{G}_{i,l_xl_yl_z,{\bf s}}  = \sum_{j} R_{nij} \prod_{\alpha={x,y,z}} r_{ij\alpha}^{l_{\alpha}}$$


## todolist
- implement a lammps interface for production runs

