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
import ase
from ase.neighborlist import neighbor_list
from tfr_data_processing import write_tfr, get_tfrs
import warnings

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
        cell = [100.0,100.0,100.0]
        #cell = None
        pbc=True
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
def prepare_and_split_data_pad(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec):
    #this should be parellize at some point   
    all_positions = []
    all_species_encoder = []
    all_energies = []
    all_forces = []
    all_natoms = []
    cells = []
    replica_idx = []
    all_C6 = []
    #get nmax atoms
    nmax = -1000000
    for file in files:
        if data_format == 'panna_json':
            atoms = convert_json2ASE_atoms(atomic_energy,file,C6_spec,species)
        elif data_format == 'ase' or data_format == 'xyz' :
            atoms = file
        nat = atoms.get_global_number_of_atoms()
        nmax = np.max([nmax,nat])

    for file in files:
        if data_format == 'panna_json':
            atoms = convert_json2ASE_atoms(atomic_energy,file,C6_spec,species)

            all_energies.append(atoms.info[energy_key])
            all_forces.append(atoms.get_array(force_key))
            nat = atoms.get_global_number_of_atoms()
            ndiff = nmax - nat
        elif data_format == 'ase' or data_format == 'xyz' :
            atoms = file
            symbols = list(atoms.symbols)
            nat = atoms.get_global_number_of_atoms()
            ndiff = nmax - nat
            
            #if not atoms.cell:
            #    atoms.cell = np.eye(3) * 100
            E0 = 0.0
            for i,sp in enumerate(species):
                Nsp = len(np.where(np.asarray(symbols).astype(str)==sp)[0])
                E0 += Nsp * atomic_energy[i]
 #           print(atoms.get_potential_energy())
            all_energies.append(atoms.get_potential_energy()-E0)
            forces = atoms.get_forces()
            forces = np.pad(forces, pad_width=((0,ndiff),(0,0)))
            all_forces.append(forces)

            _encoder = np.asarray([atomic_number(ss) for ss in symbols])
            atoms.new_array('encoder', _encoder)
            # C6 are in Ha * au^6
            to_eV = 27.211324570273 * 0.529177**6
            C6 = np.asarray([C6_spec[ss] for ss in symbols])
            atoms.new_array('C6', C6)


        positions = atoms.positions 
        positions = np.pad(positions, pad_width=((0,ndiff),(0,0)))
        all_positions.append(positions)
        encoder = atoms.get_array('encoder')
        encoder = np.pad(encoder, pad_width=(0,ndiff))
        all_species_encoder.append(encoder)
        C6 = atoms.get_array('C6')
        C6 = np.pad(C6, pad_width=(0,ndiff))

        all_C6.append(C6)
        all_natoms.append(atoms.get_global_number_of_atoms())
        cells.append(atoms.cell)
        replica_idx.append(replicas_max_idx(atoms.cell, rc, pbc=pbc))

    Ntest = int(test_fraction*len(all_natoms))

    cells_test = tf.constant(cells[:Ntest])
    cells_train = tf.constant(cells[Ntest:])
    replica_idx_test = tf.constant(replica_idx[:Ntest])
    replica_idx_train = tf.constant(replica_idx[Ntest:])

    all_positions_test = tf.constant(all_positions[:Ntest])
    all_positions_train = tf.constant(all_positions[Ntest:])

    #forces
    all_forces_test = tf.constant(all_forces[:Ntest])
    all_forces_train = tf.constant(all_forces[Ntest:])

    #print(Ntest, len(all_positions_train), len(all_positions_test))
    all_species_encoder_test = tf.constant(all_species_encoder[:Ntest], dtype=tf.float32)
    all_species_encoder_train = tf.constant(all_species_encoder[Ntest:], dtype=tf.float32)

    all_natoms_test = tf.constant(all_natoms[:Ntest])
    all_natoms_train = tf.constant(all_natoms[Ntest:])

    all_energies_test = tf.constant(all_energies[:Ntest])
    all_energies_train = tf.constant(all_energies[Ntest:])
    all_C6_test = tf.constant(all_C6[:Ntest])
    all_C6_train = tf.constant(all_C6[Ntest:])


    train_data = input_function((all_positions_train, all_species_encoder_train,
                                 all_natoms_train,cells_train, replica_idx_train, all_C6_train,
                                 all_energies_train, all_forces_train),
                                shuffle=True, batch_size=batch_size)

    test_data = input_function((all_positions_test, all_species_encoder_test,
                                all_natoms_test, cells_test,replica_idx_test, all_C6_test,
                                all_energies_test, all_forces_test),
                                shuffle=True, batch_size=batch_size)
    return train_data, test_data

def prepare_and_split_data_ragged(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec,model_dir,
                     evaluate_test):
    #this should be parellize at some point   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    all_positions = []
    all_species_encoder = []
    all_energies = []
    all_forces = []
    all_natoms = []
    cells = []
    replica_idx = []
    all_C6 = []
    all_first_atom = []
    all_second_atom = []
    all_shift_vectors = []
    all_neigh = []

    for j,file in enumerate(files):
        if data_format == 'panna_json':
            atoms = convert_json2ASE_atoms(atomic_energy,file,C6_spec,species)
            all_energies.append(atoms.info[energy_key])
            all_forces.append(atoms.get_array(force_key))
            if atoms.cell is None or np.linalg.norm(atoms.cell)<1e-6:
                atoms.set_cell(np.eye(3)*100)

            #print(atoms.cell,np.linalg.norm(atoms.cell))
        elif data_format == 'ase' or data_format == 'xyz' :
            atoms = file
            symbols = list(atoms.symbols)
            if atoms.cell is None or np.linalg.norm(atoms.cell)<1e-6:
                #atoms.cell = np.zeros((3,3))
                atoms.set_cell(np.eye(3)*100)
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
        #replica_idx.append(replicas_max_idx(atoms.cell, rc, pbc=pbc))
        replica_idx.append([0,0,0])
 #       all_species_symbols.append(atoms.get_chemical_symbols())
        
        first_atom_idx,second_atom_idx,shift_vector = neighbor_list('ijS', atoms, rc)
        

        all_first_atom.append(first_atom_idx)
        all_second_atom.append(second_atom_idx)
        all_shift_vectors.append(shift_vector)
        all_neigh.append(len(first_atom_idx))
        if j % 1000 == 0:
            print(j, len(first_atom_idx), rc)
    Ntest = int(test_fraction*len(all_natoms))
    neigh_max = np.max(all_neigh[Ntest:])
    nmax = np.max(all_natoms[Ntest:])
    nelement = 50
    all_dict = {'positions':all_positions[Ntest:], 
                'atomic_number':all_species_encoder[Ntest:], 
                'C6':all_C6[Ntest:],
                'cells':cells[Ntest:],
                'natoms':all_natoms[Ntest:],
                'i':all_first_atom[Ntest:],
                'j':all_second_atom[Ntest:],
                'S':all_shift_vectors[Ntest:],
                'nneigh':all_neigh[Ntest:],
                'energy':all_energies[Ntest:],
                'forces':all_forces[Ntest:]}
    Ntrain = len(all_natoms) - Ntest
    if Ntrain > 0:
        write_tfr('train',all_dict,
                  nmax,neigh_max, 
                  nelement=nelement, tfr_dir=model_dir+'/tfrs_train')
    else:
        warnings.warn('there are no configurations for training!')

    all_dict = {'positions':all_positions[:Ntest], 
                'atomic_number':all_species_encoder[:Ntest],
                'C6':all_C6[:Ntest],
                'cells':cells[:Ntest],
                'natoms':all_natoms[:Ntest],
                'i':all_first_atom[:Ntest],
                'j':all_second_atom[:Ntest],
                'S':all_shift_vectors[:Ntest],
                'nneigh':all_neigh[:Ntest],
                'energy':all_energies[:Ntest],
                'forces':all_forces[:Ntest]}

    # this is just to distingush test and validation
    test_dir = model_dir+'/tfrs_validate'
    if evaluate_test:
        test_dir = model_dir+'/tfrs_test'
    neigh_max = np.max(all_neigh[:Ntest])
    nmax = np.max(all_natoms[:Ntest])
    if Ntest > 0:
        write_tfr('test',all_dict, 
                  nmax,neigh_max,
                  nelement=nelement, tfr_dir=test_dir)
    else:
        warnings.warn('there are no configurations for validation!')

    ''' 
    filenames = tf.io.gfile.glob("tfrs_train/train*.tfrecords")
    train_data = get_tfrs(filenames, batch_size)
    
    filenames = tf.io.gfile.glob("tfrs_test/test*.tfrecords")
    test_data = get_tfrs(filenames, batch_size)

    all_first_atom_test = tf.ragged.constant(all_first_atom[:Ntest],dtype=tf.int32,row_splits_dtype=tf.int32)
    all_first_atom_train = tf.ragged.constant(all_first_atom[Ntest:],dtype=tf.int32,row_splits_dtype=tf.int32)
    all_neigh_train = tf.constant(all_neigh[Ntest:],dtype=tf.int32)
    all_neigh_test = tf.constant(all_neigh[:Ntest],dtype=tf.int32)
    
    all_second_atom_test = tf.ragged.constant(all_second_atom[:Ntest], dtype=tf.int32,row_splits_dtype=tf.int32)
    all_second_atom_train = tf.ragged.constant(all_second_atom[Ntest:],dtype=tf.int32,row_splits_dtype=tf.int32)
    all_shift_vectors_test = tf.ragged.constant(all_shift_vectors[:Ntest],dtype=tf.int32,row_splits_dtype=tf.int32)
    all_shift_vectors_train = tf.ragged.constant(all_shift_vectors[Ntest:],dtype=tf.int32,row_splits_dtype=tf.int32)

#    all_species_symbols_test = tf.ragged.constant(all_species_symbols[:Ntest])
#    all_species_symbols_train = tf.ragged.constant(all_species_symbols[Ntest:])

    cells_test = tf.constant(cells[:Ntest],dtype=tf.float32)
    cells_train = tf.constant(cells[Ntest:],dtype=tf.float32)
    replica_idx_test = tf.constant(replica_idx[:Ntest],dtype=tf.int32)
    replica_idx_train = tf.constant(replica_idx[Ntest:],dtype=tf.int32)

    all_positions_test = tf.ragged.constant(all_positions[:Ntest],dtype=tf.float32,row_splits_dtype=tf.int32)
    all_positions_train = tf.ragged.constant(all_positions[Ntest:],dtype=tf.float32,row_splits_dtype=tf.int32)

    #forces
    all_forces_test = tf.ragged.constant(all_forces[:Ntest],dtype=tf.float32,row_splits_dtype=tf.int32)
    all_forces_train = tf.ragged.constant(all_forces[Ntest:],dtype=tf.float32,row_splits_dtype=tf.int32)

    #print(Ntest, len(all_positions_train), len(all_positions_test))
    all_species_encoder_test = tf.ragged.constant(all_species_encoder[:Ntest], dtype=tf.float32,row_splits_dtype=tf.int32)
    all_species_encoder_train = tf.ragged.constant(all_species_encoder[Ntest:], dtype=tf.float32,row_splits_dtype=tf.int32)

    all_natoms_test = tf.constant(all_natoms[:Ntest],dtype=tf.int32)
    all_natoms_train = tf.constant(all_natoms[Ntest:],dtype=tf.int32)

    all_energies_test = tf.constant(all_energies[:Ntest],dtype=tf.float32)
    all_energies_train = tf.constant(all_energies[Ntest:],dtype=tf.float32)
    all_C6_test = tf.ragged.constant(all_C6[:Ntest],dtype=tf.float32,row_splits_dtype=tf.int32)
    all_C6_train = tf.ragged.constant(all_C6[Ntest:],dtype=tf.float32,row_splits_dtype=tf.int32)

    

    train_data = input_function((all_positions_train, all_species_encoder_train,
                                 all_natoms_train,cells_train, replica_idx_train, all_C6_train,
                                 all_first_atom_train,all_second_atom_train,all_shift_vectors_train,
                                 all_neigh_train,
                                 all_energies_train, all_forces_train),
                                shuffle=True, batch_size=batch_size)

    test_data = input_function((all_positions_test, all_species_encoder_test,
                                all_natoms_test, cells_test,replica_idx_test, all_C6_test, 
                                all_first_atom_test,all_second_atom_test,all_shift_vectors_test,
                                all_neigh_test,
                                all_energies_test, all_forces_test),
                                shuffle=True, batch_size=batch_size)
    '''
    return nmax,neigh_max

def prepare_and_split_atoms(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec):
    #this should be parellize at some point   
    all_atoms = []
    for file in files:
        if data_format == 'panna_json':
            atoms = convert_json2ASE_atoms(atomic_energy,file,C6_spec,species)
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
            #all_energies.append(atoms.get_potential_energy()-E0)
            #all_forces.append(atoms.get_forces())

            _encoder = np.asarray([atomic_number(ss) for ss in symbols])
            atoms.new_array('encoder', _encoder)
            # C6 are in Ha * au^6
            to_eV = 27.211324570273 * 0.529177**6
            C6 = np.asarray([C6_spec[ss] for ss in symbols])
            atoms.new_array('C6', C6)
            atoms.info = {'energy':atoms.get_potential_energy()-E0}

        all_atoms.append(atoms)

    Ntest = int(test_fraction*len(all_atoms))

    all_atoms_test = tf.constant(all_atoms[:Ntest])
    all_atoms_train = tf.constant(all_atoms[Ntest:])
    train_data = input_function(all_atoms_train,
                                shuffle=True, batch_size=batch_size)

    test_data = input_function(all_atoms_test,
                                shuffle=True, batch_size=batch_size)
    return train_data, test_data

def data_preparation(data_dir, species, data_format, 
                     energy_key, force_key,
                     rc, pbc, batch_size, 
                     test_fraction=0.1,
                     atomic_energy=[],
                     atomic_energy_file=None,
                     model_version='v0',
                     model_dir='tmp',
                     evaluate_test=False):
    
#    rc = np.max([rc_rad, rc_ang])

    if data_format == 'panna_json':
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.split('.')[-1]=='example']
    elif data_format == 'xyz':
        # note, this is already ase atom object
        files = read(data_dir, index=':')
    elif data_format == 'ase':
        files = data_dir
        #collect configurations
    #all_configs_ase = []

    print(f'we have a total of {len(files)} configurations')
    #shuffle dataset before splitting
    # this method shuffle files in-place

    random.Random(42).shuffle(files)

    #determine atomic zeros
    if len(atomic_energy)==0:
        print('estimating atomic reference from a linear fit')
        mat_A = []
        energy = []
        in_spec = []
        #we can also estimate the mean and std of the descriptors
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
            #if len(energy) > int(0.1*len(files)) and check == len(species):
            if len(energy) > len(species)+5 and check == len(species):
                break
        #fit

        np.random.seed(42)
        mat_A = np.asarray(mat_A).astype(float)
        energy = np.asarray(energy)
        #print(energy, mat_A)

        A = np.matmul(mat_A.T, mat_A)
        n,m = A.shape
        A += np.eye(n) * 1e-3

        b = np.matmul(mat_A.T, energy)
        atomic_energy = np.matmul(np.linalg.inv(A), b)
        print('atomic energy used are :', atomic_energy, 'mae: ', 
          np.mean(np.abs(energy-np.matmul(mat_A,atomic_energy.T))))
    
    else:
        print('atomic energy used are :', atomic_energy)

    #dump atomic energy to a json file
    E0 = {x:y for x,y in zip(species, atomic_energy)}

    if atomic_energy_file:
        with open(atomic_energy_file, 'w') as out_file:
            json.dump(E0, out_file)

    



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

    # a quick check whether to create tfrs 
    Nconf = len(files)
    ntest = int(test_fraction*Nconf)
    
    filenames = tf.io.gfile.glob(model_dir+"/tfrs_train/train*.tfrecords")
    if evaluate_test:
        filenames_test = tf.io.gfile.glob(model_dir+"/tfrs_test/test*.tfrecords")
    else:
        filenames_test = tf.io.gfile.glob(model_dir+"/tfrs_validate/test*.tfrecords")

    recomputing = 0

    if len(filenames) < (Nconf-ntest) / 50 and not evaluate_test:
        nmax,neigh_max = prepare_and_split_data_ragged(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec,model_dir,
                     evaluate_test)
        with open(model_dir+'/max_numbers.json', 'w') as out_file:
            json.dump({"nmax":int(nmax), "neigh_max":int(neigh_max)}, out_file)
        recomputing = 1
    elif evaluate_test and len(filenames_test) < ntest / 50:
        nmax,neigh_max = prepare_and_split_data_ragged(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec,model_dir,
                     evaluate_test)
        recomputing = 1
    #else:
    #    nn = json.load(open(model_dir+'/max_numbers.json'))
    #    nmax = int(nn['nmax'])
    #    neigh_max = int(nn['neigh_max'])
    if recomputing == 1: 
        filenames = tf.io.gfile.glob(model_dir+"/tfrs_train/train*.tfrecords")
        if evaluate_test:
            filenames_test = tf.io.gfile.glob(model_dir+"/tfrs_test/test*.tfrecords")
        else:
            filenames_test = tf.io.gfile.glob(model_dir+"/tfrs_validate/test*.tfrecords")

    train_data = get_tfrs(filenames, batch_size)
    test_data = get_tfrs(filenames_test, batch_size)


    '''
    if model_version == 'v0' or model_version=='linear':
        train_data, test_data = prepare_and_split_data_ragged(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec,model_dir)
    elif model_version=='ase':
        train_data, test_data = prepare_and_split_atoms(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec)
    '''
    return [train_data, test_data, species_identity]
