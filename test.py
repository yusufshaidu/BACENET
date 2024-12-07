from ase import Atoms
import ase
from ase.neighborlist import neighbor_list
from ase.io import read
with tf.GradientTape() as g:
    positions = tf.cast(atom.positions, tf.float32)
    g.watch(positions)
    atoms = Atoms(positions=positions, cell=atom.cell, pbc=True, symbols=atom.get_chemical_symbols())
    
    i, j,S = neighbor_list('ijS', atoms, 5.0)
    D = tf.gather(positions,j)-tf.gather(positions,i) + S.dot(atoms.cell)
    print(D.shape)
    d = tf.linalg.norm(D, axis=-1)
    print(d)
    C=1.0
    energy = -0.5*tf.reduce_sum(C/d**6)
forces = -g.jacobian(energy, positions)
spec = atom.get_atomic_numbers()
print(forces)
