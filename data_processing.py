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

from pbc import replicas_max_idx


def convert_json2ASE_atoms(atomic_energy, file):
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

    return symbols.index(species)

def input_function(x, shuffle=True, batch_size=32): # inner function that will be returned
    dataset = tf.data.Dataset.from_tensor_slices(x)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.ragged_batch(batch_size,drop_remainder=True) # split dataset into batch_size batches and repeat process for num_epochs
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
                     rc_rad, rc_ang, pbc, batch_size, 
                     test_fraction=0.1,
                     atomic_energy=[]):
    
    rc = np.max([rc_rad, rc_ang])

    if data_format == 'panna_json':
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.split('.')[-1]=='example']
    elif data_format == 'xyz':
        # note, this is already atom object
        files = read(data_dir, index=':')
        #collect configurations
    #all_configs_ase = []

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
    #  atoms = convert_json2ASE_atoms(files[0],species)
    #  e0 = atoms.info['energy'] / len(atoms.positions)
    #  print(e0)
    #implement multiprocessing
    #species encoder for all atomic species.
    #_spec_encoder = species_encoder()
    
    species_identity = [atomic_number(s) for s in species]

    #partial_convert = convert_json2ASE_atoms(all_species_encoder,atomic_energy)
    #number of precesses
    #p = Pool(num_process)
    #Ndata = len(files)

    #this should be parellize at some point   
    for file in files:
        if data_format == 'panna_json':
            atoms = convert_json2ASE_atoms(atomic_energy,file)
        elif data_format == 'xyz':
            atoms = file
            symbols = list(atoms.symbols)
            _encoder = np.asarray([atomic_number[ss] for ss in symbols])
            atoms.new_array('encoder', _encoder)

            
    #    if atoms.info['energy'] > 30.0:
    #        continue
        #all_configs_ase.append(atoms)
        all_energies.append(atoms.info['energy'])
        all_positions.append(atoms.positions)
        all_species_encoder.append(atoms.get_array('encoder'))
        try:
            all_forces.append(atoms.get_array('forces'))
        except:
            all_forces.append(atoms.get_forces())

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



    train_data = input_function((all_positions_train, all_species_encoder_train,
                                 all_natoms_train,cells_train, replica_idx_train,
                                 all_energies_train, all_forces_train),
                                shuffle=True, batch_size=batch_size)

    test_data = input_function((all_positions_test, all_species_encoder_test,
                                all_natoms_test, cells_test,replica_idx_test,
                                all_energies_test, all_forces_test),
                                shuffle=True, batch_size=batch_size)


    return [train_data, test_data, species_identity]
