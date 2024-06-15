#This is a dataset of 344,654 clusters of ethylene glycol EG, diethylene glycol (EG)2, and triethylene glycol EG(3), along with DFT labels at the wB97X-D3BJ/def2-TZVPD level of theory.

#The primary key for each datapoint is a string representing an index from 0 to 344,653 inclusive. Each datapoint is a dictionary containing the following keys:

#* atomicNumbers: atomic number of each atom
#* charge: net charge for each system, in elementary charges (always 0 for this dataset)
#* datasetTitle: indicates the sampling step which produced this data, out of the following six options:
#    * EG_run: initial cluster sampling of EG, (EG)2, and (EG)3
#    * decomp_EG_monomer: decomposition sampling for EG
#    * decomp_EG_dimer: decomposition sampling for (EG)2
#    * decomp_EG_trimer: decomposition sampling for (EG)3
#    * active_one: first active learning round
#    * active_two: second active learning round
#* elements: element for each atom (same info as atomicNumbers)
#* labels: for this dataset, always 'wB97X-D3BJ__def2-TZVPD', containing within the following keys:
#    * atomizationEnergy: total energy with fitted per-atom atomic energies removed, in Hartree (H: -31.91939585331449, C: -6.840292406034887, O: -12.5395523076437)
#    * dipoleMoment: DFT system dipole moment, in e-Angstroms
#    * gradient: energy gradient dE/dx (negative of atomic forces), in Hartree/Angstrom
#    * totalEnergy: total energy from wB97X-D3BJ/def2-TZVPD, in Hartree
#    * xtbCharges: xtb partial charges, in elementary charge units
#* multiplicity: spin multiplicity for each system (always 1 for this dataset)
#* positions: xyz coordinates of atomic nuclei, in Angstroms



import os, sys, json
import numpy as np
from ase import Atoms
from ase.io import write

infile = sys.argv[1]
outdir = sys.argv[2]

def get_E0(species):
    atomic_energies = {'H': -31.91939585331449, 'C': -6.840292406034887, 'O': -12.5395523076437}
    E0 = 0
    for sp in atomic_energies.keys():
        nsp = len(np.where(np.asarray(species).astype(str)==sp)[0])
        E0 += nsp * atomic_energies[sp]
    return E0


if not os.path.exists(outdir):
    os.mkdir(outdir)
if not os.path.exists('EG_test_json'):
    os.mkdir('EG_test_json')

data = json.load(open(infile))
Nstr = len(data.keys())

print(Nstr)

all_forces = [[],[],[]]
all_energy = []
for key in data.keys():
    if int(key) > 0:
        break
    info = {}
    d = data[key]
    total_charge = d['charge']
    symbols = d['elements']
    positions = d['positions']
    atoms = Atoms(positions=positions, cell=[20,20,20], symbols=symbols)

    write('test.xyz', atoms)

    name = d['datasetTitle']
    cohesive_E = d['labels']['wB97X-D3BJ__def2-TZVPD']['atomizationEnergy']
    total_energy = d['labels']['wB97X-D3BJ__def2-TZVPD']['totalEnergy']

    dipole_moment = d['labels']['wB97X-D3BJ__def2-TZVPD']['dipoleMoment']
    charges = d['labels']['wB97X-D3BJ__def2-TZVPD']['xtbCharges']
    
    forces = d['labels']['wB97X-D3BJ__def2-TZVPD']['gradient']
    
    all_energy.append(cohesive_E)
    for i in range(3):
        all_forces[i] = np.append(all_forces[i], np.asarray(forces)[:,i])

    info['total_charge'] = total_charge
    nat = len(positions)
    info['atoms'] = [[idx, sym, pos, [-force[0],-force[1],-force[2]], charge] for idx, sym, pos, force, charge in zip(range(1,nat+1), symbols, positions, forces, charges)]
    info['energy'] = [cohesive_E, 'Ha']
    info['total_energy'] = [total_energy, 'Ha']
    info['atomic_position_unit'] = 'cartesian'
    info['unit_of_length'] = 'angstrom'

    info['lattice_vectors'] = [[0,0,0],[0,0,0],[0,0,0]]

    filename = f"{key}_{name}.example"
    out_file = open(os.path.join(outdir, filename), "w")
    json.dump(info, out_file)
    E0 = get_E0(symbols)

    out_file.close()
    if int(key) % 100==0:
        out_file = open(os.path.join('EG_test_json', filename), "w")
        json.dump(info, out_file)
        print(key, Nstr, f"done {Nstr-int(key)}", total_energy-E0-cohesive_E)
    out_file.close()
    #print(key)

np.savetxt('all_energy-EG.dat', all_energy)
np.savetxt('all_forces-EG.dat', np.stack([all_forces[0], all_forces[1], all_forces[2]]).T)
import matplotlib.pyplot as plt
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,12))
ax1.hist(np.asarray(all_energy), bins='auto', density=True)
ax1.set_xlabel('Energy (Ha)')
ax1.set_ylabel('Distribution')
axes = ['x','y','z']
for i in range(3):
    ax2.hist(all_forces[i], bins='auto', density=True, label=f'f{axes[i]}')
ax2.set_xlabel('Forces (Ha/Ang)')
ax2.set_ylabel('Distribution')

plt.savefig('Distribution_EG.png')
plt.show()

