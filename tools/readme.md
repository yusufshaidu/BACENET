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

$$G_{\bf s}^3 = \frac{2}{2^{\zeta}}\sum_{l=0}^{\zeta}\lambda^l G_{{\bf s}l}^3$$

where

$$G_{{\bf s}l}^3 =\sum_{l_xl_yl_z} \frac{\zeta!}{l!(\zeta-l)!} \frac{l!}{l_x!l_y!l_z!} (G_{i,l_xl_yl_z,{\bf s}})^2$$

and 

$$G_{i,l_xl_yl_z,{\bf s}}  = \sum_{j} R_{s,ij} \prod_{\alpha={x,y,z}} r_{ij\alpha}^{l_{\alpha}}$$

where $${\bf s} =(n,s)$$

There are two possible implementation of this descriptors:
- following the original implementation of Behler Parrinello descriptors and chosing different $\zeta$ and performing the sum explicitly
- Using a fixed zeta and each component of zeta as a different channel. This is much computationally efficient. In this case, the normalization factor $\frac{2}{2^{\zeta}}$ is irrelevant and can be dropped, as well as the $\lambda$

Both of these are implemented 
## Higher body-order terms
Following the same form for the three body terms, we can write a four-body descriptors as

$$G_i^{(4)}(s) = \sum_{jkm} (\frac{1+\lambda_s {\rm cos}(\theta_{ijk})}{2})^\zeta (\frac{1+\lambda_s {\rm cos}(\theta_{ijm})}{2})^\zeta R_{ijs} R_{iks} R_{ijm}$$

Other combination of jkm neighbors of i can be used but they all gives similar expression. Expanding in powers of z, and also expanding $cos\theta_{ijk}$ terms, we got
$$G_{\bf s}^3 = \sum_{l_1l_2}^{\zeta}\lambda^l_1\lambda^l_2\frac{\zeta!}{l_1!(\zeta-l_1)!} \frac{\zeta!}{l_2!(\zeta-l_2)!}\sum_{l_{1x}l_{1y}l_{1z}} \sum_{l_{2x}l_{2y}l_{2z}} \frac{l_1!}{l_{1x}!l_{1y}!l_{1z}!} \frac{l_2!}{l_{2x}!l_{2y}!l_{2z}!} (G_{i,l_{1x}l_{1y}l_{1z},{\bf s}} \otimes G_{i,l_{2x}l_{2y}l_{2z},{\bf s}}) G_{i,l_{12x}l_{12y}l_{12z},{\bf s}}$$

where $$G_{i,l_{12x}l_{12y}l_{12z},{\bf s}}=\sum_{j} R_{s,ij} \prod_{\alpha={x,y,z}} r_{ij\alpha}^{l_{1\alpha}} \otimes \prod_{\alpha={x,y,z}} r_{ij\alpha}^{l_{2\alpha}}=\sum_{j} R_{s,ij} \prod_{\alpha={x,y,z}} r_{ij\alpha}^{l_{1\alpha}+l_{2\alpha}}$$

As before, we drop the sum over, we decomposed the descriptors into $l_1 \times l_2$ components

###
## todolist
- implement a lammps interface for production runs
- implement graph based model using the $G_{i,l_xl_yl_z,{\bf s}}$ as the initial node features
