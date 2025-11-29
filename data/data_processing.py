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
from data.tfr_data_processing import write_tfr, get_tfrs
from models.ewald import ewald
import warnings

from concurrent.futures import ProcessPoolExecutor, as_completed


def compute_electrostatic_energy0(cell, positions, gaussian_width, pbc,
                                  shelld, atomic_charges, nuclei_charges):
    _ewald = ewald(positions, cell, len(positions),
                        gaussian_width,accuracy=1e-3,
                               gmax=None, pbc=pbc,
                               )
    #Vij = _ewald.recip_space_term() if pbc else _ewald.real_space_term()
    #Vij = Vij.numpy()
    q_outer = atomic_charges[:,None] * atomic_charges[None,:]
    qz_outer = atomic_charges[:,None] * nuclei_charges[None,:]
    zz_outer = nuclei_charges[:,None] * nuclei_charges[None,:]
    
    Vij, Vij_qz, Vij_zz = _ewald.recip_space_term_with_shelld(shelld)
    return 0.5 * (np.sum(Vij.numpy() * q_outer + 
                        Vij_qz.numpy() * qz_outer + 
                        Vij_zz.numpy() * zz_outer) + np.sum(0.1 * shelld**2))

#from ml_potentials.pbc import replicas_max_idx
#try:
#    from pbc import replicas_max_idx
#except:
#    from .pbc import replicas_max_idx
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

def get_energy_ase(atom, species,energy_key):
    Ry2eV = 13.6057039763
    symbols = list(atom.symbols)

    nspec = []
    in_spec = []
    for i,sp in enumerate(species):
        Nsp = len(np.where(np.asarray(symbols).astype(str)==sp)[0])
        if Nsp > 0:
            in_spec.append(sp)

        nspec.append(Nsp)
    try:
        energy = atom.get_potential_energy()
    except:
        energy = atom.info[energy_key]
#    print(energy)
    return [nspec, in_spec, energy]

def convert_json2ASE_atoms(atomic_energy, file, C6_spec, species):
    Ry2eV = 13.6057039763
    data = json.load(open(file))
    charges = None
    try:
        idx, symbols, positions, forces = zip(*data['atoms'])
    except:
        idx, symbols, positions, forces, charge = zip(*data['atoms'])
    if charges is None:
        charges = np.zeros(len(positions))
    else:
        charges = np.array(charges)
    
    total_charge = np.sum(charges)

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
    atoms.new_array('charges', charges)
    atoms.new_array('encoder',_spec_encoder)
    atoms.new_array('C6',C6)
    atoms.info = {'energy':energy-E0, 'total_charge':total_charge}

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

def process_ase(atoms, evaluate_test,species,atomic_energy,C6_spec,energy_key,force_key):
        # Ensure a nonzero box
        if atoms.cell is None or np.linalg.norm(atoms.cell) < 1e-6:
            atoms.set_cell(np.eye(3) * 100)

        # Zero‐point energy correction
        E0 = sum(
            list(atoms.symbols).count(sp) * atomic_energy[i]
            for i, sp in enumerate(species)
        )

        if evaluate_test >= 0:
            try:
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                charges = atoms.get_initial_charges()
            except:
                energy = atoms.info[energy_key]
                forces = atoms.get_array(force_key)
                charges = atoms.get_initial_charges()

            energy -= E0
        else:
            energy = 0.0
            forces = np.zeros((len(atoms.positions), 3))
            charges = np.zeros(len(atoms.positions))
        # Atomic encodings
        encoder = atoms.get_atomic_numbers()
        atoms.set_array('encoder', encoder)
        atoms.set_array('charges', charges)


        # C6 coefficients
        C6_vals = np.array([C6_spec[z] for z in atoms.get_chemical_symbols()])
        atoms.set_array('C6', C6_vals)

        return atoms, energy, forces, np.sum(charges)

def _process_file(args):
    """
    Worker function: loads one file, computes all the arrays, returns a dict.
    """
    file, data_format, species, atomic_energy, C6_spec, energy_key, force_key, rc, evaluate_test,covalent_radii = args

    # 1) Load ASE Atoms object & compute energy/forces
    if data_format == 'panna_json':
        atoms = convert_json2ASE_atoms(atomic_energy, file, C6_spec, species)
        energy = atoms.info[energy_key]
        forces = atoms.get_array(force_key)
        charges = atoms.get_array('charges')
        total_charge =  atoms.info['total_charge']
    else:
        atoms = file
        atoms, energy, forces, total_charge = process_ase(atoms, evaluate_test, species,atomic_energy,C6_spec, energy_key, force_key)

    gaussian_width = np.array([covalent_radii[x] for x in atoms.get_chemical_symbols()])
    
    # Ensure box
    if atoms.cell is None or np.linalg.norm(atoms.cell) < 1e-6:
        atoms.set_cell(np.eye(3) * 100)

    # Build neighbor list
    i_list, j_list, shifts = neighbor_list('ijS', atoms, rc)

    return {
        'positions': atoms.positions,
        'cells':      atoms.cell,
        'atomic_number':   atoms.get_array('encoder'),
        'C6':        atoms.get_array('C6'),
        'gaussian_width': gaussian_width,
        'energy':    energy,
        'forces':    forces,
        'natoms':    atoms.get_global_number_of_atoms(),
        'i':     i_list,
        'j':    j_list,
        'S':    shifts,
        'nneigh':    len(i_list),
        'total_charge': total_charge,
        'charges': atoms.get_array('charges')
    }

def load_structure_data_parallel(files, data_format, species, atomic_energy,
                                 C6_spec, energy_key, force_key, rc,
                                 evaluate_test, max_workers=4):
    """
    Parallel version of load_structure_data.
    """
    covalent_radii = {x:element(x).covalent_radius*0.01 for x in species}
    if evaluate_test < 0: # this is inference basically with ASE

        args = (files[0], data_format, species, atomic_energy, C6_spec, energy_key, force_key, rc, evaluate_test,covalent_radii)
        data = _process_file(args)
        _data = {k:[] for k in data.keys()}
        for k, v in data.items():
            _data[k].append(v)
        return _data

    #print(files, data_format, species, atomic_energy, C6_spec, energy_key, force_key, rc, evaluate_test,covalent_radii)
    args_iter = (
        (f, data_format, species, atomic_energy, C6_spec, energy_key, force_key, rc, evaluate_test,covalent_radii)
        for f in files
    )
    #print(_process_file(args_iter[0]))
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = [exe.submit(_process_file, args) for args in args_iter]
        for fut in as_completed(futures):
            results.append(fut.result())

    # Now unzip the list of dicts into arrays of lists
    data = {k: [] for k in results[0].keys()}
    for res in results:
        for k, v in res.items():
            data[k].append(v)

    return data

def prepare_and_split_data_ragged(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec,model_dir,
                     evaluate_test,max_workers):
    #this should be parellize at some point   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    data = load_structure_data_parallel(files, data_format, species, atomic_energy,
                                 C6_spec, energy_key, force_key, rc,
                                 evaluate_test, max_workers=max_workers)
    nconfigs = len(data['natoms'])
    Ntest = int(test_fraction*len(data['natoms']))
    
    nelement = 50

    Ntrain = nconfigs - Ntest
    if Ntrain > 0:
        #neigh_max = np.max(all_neigh[Ntest:])
        #nmax = np.max(all_natoms[Ntest:])
        neigh_max = np.max(data['nneigh'][Ntest:])
        nmax = np.max(data['natoms'][Ntest:])
        all_dict = {}
        for key,val in data.items():
            all_dict[key] = val[Ntest:]
        write_tfr('train',all_dict,
                  nmax,neigh_max, 
                  nelement=nelement, tfr_dir=model_dir+'/tfrs_train')
    #else:
    #    warnings.warn('there are no configurations for training!')
    if Ntest > 0:
        all_dict = {}
        for key,val in data.items():
            all_dict[key] = val[:Ntest]
        # this is just to distingush test and validation
        test_dir = model_dir+'/tfrs_validate'
        if evaluate_test == 1:
            test_dir = model_dir+'/tfrs_test'
        neigh_max = np.max(data['nneigh'][:Ntest])
        nmax = np.max(data['natoms'][:Ntest])
        write_tfr('test',all_dict, 
                  nmax,neigh_max,
                  nelement=nelement, tfr_dir=test_dir)
    #else:
    #    warnings.warn('there are no configurations for validation!')

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
            try:
                energy = atoms.get_potential_energy()
            except:
                energy = atoms.info[energy_key]
            atoms.info = {'energy':energy-E0}

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
                     evaluate_test=0,max_workers=8,
                     n_epochs=64,
                     oxidation_states={},
                     nuclei_charges={}):
    
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

    species_identity = [atomic_number(s) for s in species]
    # C6 are in Ha * au^6
    to_eV = 27.211324570273 * 0.529177**6
    C6_spec = {ss:element(ss).c6_gb * to_eV for ss in species}
    covalent_radii = {x:element(x).covalent_radius*0.01 for x in species}

    #print(f'we have a total of {len(files)} configurations')
    #shuffle dataset before splitting
    # this method shuffle files in-place
    n_samples = int((1-test_fraction) * len(files))
    repeat_count = n_samples // batch_size * n_epochs
    random.Random(42).shuffle(files)

    #determine atomic zeros from a linear fit
    # we should remove the energy due to the atomic charges especially in the cae of polarizable Qeq
    if len(atomic_energy)==0:
        print('estimating atomic reference from a linear fit')
        mat_A = []
        energy = []
        in_spec = []
        #we can also estimate the mean and std of the descriptors
        for file in files:
            if data_format == 'panna_json':
                Nspec, spec, ene = get_energy_json(file,species)
                atoms = convert_json2ASE_atoms(atomic_energy, file, C6_spec, species)
            else:
                Nspec, spec, ene =  get_energy_ase(file,species,energy_key)
                atoms = file
            '''
            shelld = np.ones_like(atoms.positions) * 0.08 # uniform displacement
            atomic_charges = np.array([oxidation_states[s] for s in atoms.get_chemical_symbols()])
            z_charges = np.array([nuclei_charges[s] for s in atoms.get_chemical_symbols()])
            gaussian_width = np.array([covalent_radii[s] for s in atoms.get_chemical_symbols()])
            Eele = compute_electrostatic_energy0(atoms.cell, atoms.positions, 
                                                 gaussian_width, pbc,
                                                 shelld, atomic_charges, z_charges)
          
            #print('electrostatic energy',Eele)
            '''
            mat_A.append(Nspec)
            energy.append(ene)
            in_spec = np.append(in_spec, spec)
            check = 0
            for s in species:
                if s in in_spec:
                    check += 1
            #if len(energy) > int(0.1*len(files)) and check == len(species):
            if len(energy) > len(species)+50 and check == len(species):
                break
        #fit

        np.random.seed(42)
        mat_A = np.asarray(mat_A).astype(float)
        energy = np.asarray(energy)
        #print(energy, mat_A)

        A = np.matmul(mat_A.T, mat_A)
        n,m = A.shape
        A += np.eye(n) * 1e-6

        b = np.matmul(mat_A.T, energy)
        atomic_energy = np.matmul(np.linalg.inv(A), b)
        print('atomic energy used are :', atomic_energy, 'mae: ', 
          np.mean(np.abs(energy-np.matmul(mat_A,atomic_energy.T))))
    
    else:
        #print('atomic energy used are :', atomic_energy)
        pass

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
    
    
    # a quick check whether to create tfrs 
    Nconf = len(files)
    ntest = int(test_fraction*Nconf)
    
    filenames = tf.io.gfile.glob(model_dir+"/tfrs_train/train*.tfrecords")
    if evaluate_test == 1:
        filenames_test = tf.io.gfile.glob(model_dir+"/tfrs_test/test*.tfrecords")
    else:
        filenames_test = tf.io.gfile.glob(model_dir+"/tfrs_validate/test*.tfrecords")

    recomputing = 0

    if len(filenames) < (Nconf-ntest) / 50 and evaluate_test not in [-1,1]:

        nmax,neigh_max = prepare_and_split_data_ragged(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec,model_dir,
                     evaluate_test,max_workers)
        with open(model_dir+'/max_numbers.json', 'w') as out_file:
            json.dump({"nmax":int(nmax), "neigh_max":int(neigh_max)}, out_file)
        recomputing = 1
    elif evaluate_test == 1 and len(filenames_test) < ntest / 50:
        nmax,neigh_max = prepare_and_split_data_ragged(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec,model_dir,
                     evaluate_test,max_workers)
        recomputing = 1
    elif evaluate_test == -1:
    #    print('computing')
        nmax,neigh_max = prepare_and_split_data_ragged(files, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction,
                     atomic_energy,C6_spec,model_dir,
                     evaluate_test,max_workers)
        recomputing = 1

    #else:
    #    nn = json.load(open(model_dir+'/max_numbers.json'))
    #    nmax = int(nn['nmax'])
    #    neigh_max = int(nn['neigh_max'])
    if recomputing == 1:
     #   print('reading')
        filenames = tf.io.gfile.glob(model_dir+"/tfrs_train/train*.tfrecords")
        if evaluate_test == 1:
            filenames_test = tf.io.gfile.glob(model_dir+"/tfrs_test/test*.tfrecords")
        if evaluate_test == 0:
            filenames_test = tf.io.gfile.glob(model_dir+"/tfrs_validate/test*.tfrecords")

    train_data = get_tfrs(filenames, batch_size, repeat_count)
    test_data = get_tfrs(filenames_test, batch_size, repeat_count)


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
    return [train_data, test_data, species_identity, n_samples]
