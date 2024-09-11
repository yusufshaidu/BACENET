from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import mendeleev
from mendeleev import element
import math 
import itertools, os
from ase.io import read, write

import argparse
from multiprocessing import Pool
from functools import partial
import random
from ase import Atoms
import json

#from ml_potentials.pbc import replicas_max_idx
try:
    from pbc import replicas_max_idx
except:
    from .pbc import replicas_max_idx
def get_energy_json(file, species):
    Ry2eV = 13.6057039763
    data = json.load(open(file))
    try:
        idx, symbols, positions, forces = zip(*data['atoms'])
    except:
        idx, symbols, positions, forces, charge = zip(*data['atoms'])
    nspec = []
    in_spec = []
    for i,sp in enumerate(species):
        Nsp = len(np.where(np.asarray(symbols).astype(str)==sp)[0])
        if Nsp > 0:
            in_spec.append(sp)

        nspec.append(Nsp)
    energy = data['energy'][0]
    if data['energy'][1] in ['Ry', 'ry', 'RY', 'ryd', 'Ryd', 'Rydberg', 'RYDBERG']:
        energy *= Ry2eV
        forces *= Ry2eV
    elif data['energy'][1] in ['Ha', 'HA', 'ha', 'Hartree', 'HARTREE']:
        energy *= (Ry2eV * 2)
    return [nspec, in_spec, energy]

def get_energy_ase(atom, species):
    Ry2eV = 13.6057039763
    symbols = list(atom.symbols)

    nspec = []
    in_spec = []
    for i,sp in enumerate(species):
        Nsp = len(np.where(np.asarray(symbols).astype(str)==sp)[0])
        if Nsp > 0:
            in_spec.append(sp)

        nspec.append(Nsp)
    energy = atom.get_potential_energy()
#    print(energy)
    return [nspec, in_spec, energy]

def convert_json2ASE_atoms(atomic_energy, file, C6_spec, species):
    Ry2eV = 13.6057039763
    data = json.load(open(file))
    try:
        idx, symbols, positions, forces = zip(*data['atoms'])
    except:
        idx, symbols, positions, forces, charge = zip(*data['atoms'])

    try:
        cell = np.asarray(data['lattice_vectors'])
        pbc = True
    except:
        #cell = np.array([[100.,0,0], [0,100,0.], [0,0,100.]])
        cell = None
        pbc=False
    E0 = 0.0
    if len(atomic_energy)>0:
        for i,sp in enumerate(species):
            Nsp = len(np.where(np.asarray(symbols).astype(str)==sp)[0])
            E0 += Nsp * atomic_energy[i]
   # print(E0)

    positions = np.asarray(positions).astype(float)
    forces = np.asarray(forces).astype(float)
#    encoder = all_species_encoder
    _spec_encoder = np.asarray([atomic_number(ss) for ss in symbols])


    C6 = np.asarray([C6_spec[ss] for ss in symbols])

    unitL = data['unit_of_length']

    if unitL in ['bohr', 'BOHR', 'Bohr']:
        if cell:
            cell *= 0.529177
        #else:
        #    cell =
        positions *= 0.529177
        forces /= 0.529177

    pos_unit = data['atomic_position_unit']
    if pos_unit in ['CRYSTAL', 'crystal', 'Crystal']:

        #_positions = []
        #for pos in positions:
        _positions = np.dot(positions, cell)
    else:
        _positions = positions.copy()

    atoms = Atoms(positions=_positions, cell=cell, symbols=symbols, pbc=pbc)

    energy = data['energy'][0]
    if data['energy'][1] in ['Ry', 'ry', 'RY', 'ryd', 'Ryd', 'Rydberg', 'RYDBERG']:
        energy *= Ry2eV
        forces *= Ry2eV
    elif data['energy'][1] in ['Ha', 'HA', 'ha', 'Hartree', 'HARTREE']:
        energy *= (Ry2eV * 2)
        forces *= (Ry2eV * 2)

    atoms.new_array('forces', forces)
    atoms.new_array('encoder',_spec_encoder)
    atoms.new_array('C6',C6)
    atoms.info = {'energy':energy-E0}

    return atoms
def atomic_number(species):
    symbols = [ 'H',                               'He',
                'Li','Be', 'B', 'C', 'N', 'O', 'F','Ne',
                'Na','Mg','Al','Si', 'P', 'S','Cl','Ar',
                 'K','Ca','Sc','Ti', 'V','Cr','Mn',
                          'Fe','Co','Ni','Cu','Zn',
                          'Ga','Ge','As','Se','Br','Kr',
                'Rb','Sr', 'Y','Zr','Nb','Mo','Tc',
                          'Ru','Rh','Pd','Ag','Cd',
                          'In','Sn','Sb','Te', 'I','Xe',
                'Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd',
                               'Tb','Dy','Ho','Er','Tm','Yb','Lu',
                               'Hf','Ta', 'W','Re','Os',
                          'Ir','Pt','Au','Hg',
                          'Tl','Pb','Bi','Po','At','Rn',
                'Fr','Ra','Ac','Th','Pa',' U','Np','Pu',
                'Am','Cm','Bk','Cf','Es','Fm','Md','No',
                'Lr','Rf','Db','Sg','Bh','Hs','Mt','Ds','Rg','Cn',
                'Nh','Fl','Mc','Lv','Ts','Og']

    return symbols.index(species) + 1

def input_function(x, shuffle=True, batch_size=32): # inner function that will be returned
    dataset = tf.data.Dataset.from_tensor_slices(x)
 #   dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(64)

    dataset = dataset.ragged_batch(batch_size,drop_remainder=False) # split dataset into batch_size batches and repeat process for num_epochs
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

#def input_function_examples(data_path, shuffle=True, batch_size=32): # inner function that will be returned
#
#    dataset = tf.data.Dataset.list_files(data_path+'*example')
#    #dataset=dataset.shuffle(1000).batch(8).repeat(32)
#    if shuffle:
#        dataset = dataset.shuffle(1000)
#    dataset = dataset.batch(batch_size).repeat(num_epochs) # split dataset into batch_size batches and repeat process for num_epochs
#    return dataset

def data_preparation(data_dir, species, data_format, 
                     energy_key, force_key,
                     rc, pbc, batch_size, 
                     test_fraction=0.1,
                     atomic_energy=[],
                     atomic_energy_file=None):
    
#    rc = np.max([rc_rad, rc_ang])

    if data_format == 'panna_json':
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.split('.')[-1]=='example']
    elif data_format == 'xyz':
        # note, this is already atom object
        files = read(data_dir, index=':')
    elif data_format == 'ase':
        files = data_dir
        #collect configurations
    #all_configs_ase = []

    print(f'we have a total of {len(files)} configurations')
    #shuffle dataset before splitting
    # this method shuffle files in-place

    random.Random(42).shuffle(files)

    all_positions = []
    all_species_encoder = []
    all_energies = []
    all_forces = []
    all_natoms = []
    cells = []
    replica_idx = []
    all_C6 = []

    #determine atomic zeros
    if len(atomic_energy)==0:
        print('estimating atomic reference from a linear fit')
        mat_A = []
        energy = []
        in_spec = []

        for file in files:
            if data_format == 'panna_json':
                Nspec, spec, ene = get_energy_json(file,species)
            else:
                Nspec, spec, ene =  get_energy_ase(file,species)

          
            mat_A.append(Nspec)
            energy.append(ene)
            in_spec = np.append(in_spec, spec)
            check = 0
            for s in species:
                if s in in_spec:
                    check += 1
            if len(energy) > len(species) + 3 and check == len(species):
                break
        #fit

        np.random.seed(42)
        mat_A = np.asarray(mat_A).astype(float)
        energy = np.asarray(energy)

        A = np.matmul(mat_A.T, mat_A)
        A += np.random.normal(scale=1e-9, size=A.shape)
        b = np.matmul(mat_A.T, energy)
        atomic_energy = np.matmul(np.linalg.inv(A), b)
    #dump atomic energy to a json file
    E0 = {x:y for x,y in zip(species, atomic_energy)}

    if atomic_energy_file:
        with open(atomic_energy_file, 'w') as out_file:
            json.dump(E0, out_file)

    print('atomic energy used are :', atomic_energy)
    



    #  atoms = convert_json2ASE_atoms(files[0],species)
    #  e0 = atoms.info['energy'] / len(atoms.positions)
    #  print(e0)
    #implement multiprocessing
    #species encoder for all atomic species.
    #_spec_encoder = species_encoder()
    
    species_identity = [atomic_number(s) for s in species]
    # C6 are in Ha * au^6
    to_eV = 27.211324570273 * 0.529177**6
    C6_spec = {ss:element(ss).c6_gb * to_eV for ss in species}

    #partial_convert = convert_json2ASE_atoms(all_species_encoder,atomic_energy)
    #number of precesses
    #p = Pool(num_process)
    #Ndata = len(files)

    #this should be parellize at some point   
    for file in files:
        if data_format == 'panna_json':
            atoms = convert_json2ASE_atoms(atomic_energy,file,C6_spec,species)
            all_energies.append(atoms.info[energy_key])
            all_forces.append(atoms.get_array(force_key))
        elif data_format == 'ase' or data_format == 'xyz' :
            atoms = file
            symbols = list(atoms.symbols)
            #if not atoms.cell:
            #    atoms.cell = np.eye(3) * 100
            E0 = 0.0
            for i,sp in enumerate(species):
                Nsp = len(np.where(np.asarray(symbols).astype(str)==sp)[0])
                E0 += Nsp * atomic_energy[i]
 #           print(atoms.get_potential_energy())
            all_energies.append(atoms.get_potential_energy()-E0)
            all_forces.append(atoms.get_forces())

            _encoder = np.asarray([atomic_number(ss) for ss in symbols])
            atoms.new_array('encoder', _encoder)
            # C6 are in Ha * au^6
            to_eV = 27.211324570273 * 0.529177**6
            C6 = np.asarray([C6_spec[ss] for ss in symbols])
            atoms.new_array('C6', C6)


            
        all_positions.append(atoms.positions)
        all_species_encoder.append(atoms.get_array('encoder'))
        all_C6.append(atoms.get_array('C6'))
        all_natoms.append(atoms.get_global_number_of_atoms())
        cells.append(atoms.cell)
        replica_idx.append(replicas_max_idx(atoms.cell, rc, pbc=pbc))

    Ntest = int(test_fraction*len(all_natoms))

    cells_test = tf.constant(cells[:Ntest])
    cells_train = tf.constant(cells[Ntest:])
    replica_idx_test = tf.constant(replica_idx[:Ntest])
    replica_idx_train = tf.constant(replica_idx[Ntest:])

    all_positions_test = tf.ragged.constant(all_positions[:Ntest])
    all_positions_train = tf.ragged.constant(all_positions[Ntest:])

    #forces
    all_forces_test = tf.ragged.constant(all_forces[:Ntest])
    all_forces_train = tf.ragged.constant(all_forces[Ntest:])

    #print(Ntest, len(all_positions_train), len(all_positions_test))
    all_species_encoder_test = tf.ragged.constant(all_species_encoder[:Ntest], dtype=tf.float32)
    all_species_encoder_train = tf.ragged.constant(all_species_encoder[Ntest:], dtype=tf.float32)

    all_natoms_test = tf.constant(all_natoms[:Ntest])
    all_natoms_train = tf.constant(all_natoms[Ntest:])

    all_energies_test = tf.constant(all_energies[:Ntest])
    all_energies_train = tf.constant(all_energies[Ntest:])
    all_C6_test = tf.ragged.constant(all_C6[:Ntest])
    all_C6_train = tf.ragged.constant(all_C6[Ntest:])


    train_data = input_function((all_positions_train, all_species_encoder_train,
                                 all_natoms_train,cells_train, replica_idx_train, all_C6_train,
                                 all_energies_train, all_forces_train),
                                shuffle=True, batch_size=batch_size)

    test_data = input_function((all_positions_test, all_species_encoder_test,
                                all_natoms_test, cells_test,replica_idx_test, all_C6_test,
                                all_energies_test, all_forces_test),
                                shuffle=True, batch_size=batch_size)


    return [train_data, test_data, species_identity]
