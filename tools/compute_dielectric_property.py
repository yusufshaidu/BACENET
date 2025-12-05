import ase
import ase.md
from ase.md import MDLogger
from bacenet.ase_interface import bacenet_Calculator 
from ase.optimize import BFGS, LBFGS
from mendeleev import element
from ase.io import read,write
import matplotlib.pyplot as plt
from ase import Atoms
from ase.filters import UnitCellFilter
from ase.optimize.sciopt import SciPyFminCG
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # only needed for 3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import sys

pi = 3.141592653589793
#reading input from command line
atoms = ase.io.read(sys.argv[1])
eps=float(sys.argv[2])
nc = int(sys.argv[3])
nc1 = nc
config_yaml = sys.argv[4]
pref = sys.argv[5]
central_atom_id = int(sys.argv[6])


if central_atom_id not in atoms.get_atomic_numbers():
    raise (f'central atomic number = {central_atom_id} is not in this configuration: set a valid atomic number')

_pcalc = bacenet_Calculator(config=config_yaml,efield=tf.ones(3) * 1e-6, 
                            central_atom_id=central_atom_id)

#set calculator
atoms.calc = _pcalc

ucf = UnitCellFilter(atoms)
init_forces = atoms.get_forces()
cell0 = atoms.cell
a0 = np.linalg.norm(cell0, axis=-1)[0] 

dyn = BFGS(ucf, maxstep=0.02)
dyn.run(fmax=0.01)

print(np.linalg.norm(init_forces-atoms.get_forces()))
print('cell after and before minimization', atoms.cell, cell0)

#display the results
print(atoms.calc.results)

#charges = myconf.get_charges()

write('check.vasp', atoms)

efield = np.ones(3) * eps

for j, e in enumerate(efield):
    _efield = np.zeros(3)
    _efield[j] = e
    #want to compute in a supercell

    rep = [1,1,1]
    for k in range(3):
        if k != j:
            rep[k] = nc1
        else:
            rep[k] = nc

    config = Atoms(cell=atoms.cell, positions=atoms.positions, symbols=atoms.symbols, pbc=True)
    
    config = config.repeat(rep)
    
    #before field aplied
    _pcalc = bacenet_Calculator(config=config_yaml,efield=np.ones(3) * 1e-6,central_atom_id=central_atom_id)
    config.calc = _pcalc
    fplus = config.get_forces()
    P0 = config.calc.results['Pi_a']

    pcalc = bacenet_Calculator(config=config_yaml,efield=_efield,central_atom_id=central_atom_id)
    _config = config.copy()
    _config.calc = pcalc
    fplus = _config.get_forces()
    chargesp = _config.get_charges()
    results = _config.calc.results
    Pi_a_plus = results['Pi_a'] - P0
    epsilon_infty = results['epsilon']
    Zstar = results['Zstar']
    #print(results['Zstar'], results['epsilon'])



    eps_0 = 8.854e-12 # F/m
    e_charge = 1.602e-19 # C
    Angs = 1.e-10 # m
    factor =  e_charge / (Angs * eps_0)
    Pi_a_plus *= factor

    print('change in polarization',Pi_a_plus)

    epsilon_infty = np.reshape(epsilon_infty, [1,-1])
    Zstar_epsilon = np.concatenate([epsilon_infty, np.reshape(Zstar, [-1,9])])
    np.savetxt(f'{pref}_{nc}x{nc1}_{eps}_zstar_epsilon_{j}.dat', Zstar_epsilon, fmt='%.6f')
