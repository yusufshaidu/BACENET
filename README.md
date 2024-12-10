## ML-Potentials using learnable species embedding and bessels radial functions (Under development)

Implementing the modified Behler Parrinello symmentry functions with learnable species embedding, radial bessels functions [https://arxiv.org/abs/2003.03123] and the three body angular centers of the descriptors.
The species embedding starts with a one-hot encoding of species and pass it through a single layer perceptron of size 64 and returns species encoder with user specified dimension.
All weights are learn from data. Neighborlist are computed with ASE and stored alongside the positions, atomic numbers, C6, nats, cells, first neighbor index(=i), second neighbor inde (=j), the shift vectors (=S), energy and forces.

## Equations
 # Redial basis functions
 $$
 R(r) = \sqrt{\frac{2}{r_c}} \frac{sin(\frac{n \pi k_n}{r_c} r)}{r} f_c{r,r_c}
 $$
## todolist
- (priority 1) implement ASE interface
- (priority) implement van der Waals and long range electrostatics!!
- implement a lammps interface for production runs
- Implement input normalization
- start from known species embedding rather than a one-hot-encoding [https://github.com/WMD-group/ElementEmbeddings] 



