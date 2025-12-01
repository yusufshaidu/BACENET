import numpy as np
import tensorflow as tf
import mendeleev
from mendeleev import element
import math 
import itertools, os
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
from networks.networks import Networks
import functions.helping_functions as help_fn
from functions.tf_linop import AOperator
from models.descriptors import *
from models.coulomb_functions import (_compute_Aij, _compute_Fia, _compute_Fiajb, 
                                      _compute_charges_disp, _compute_shell_disp_qqdd2, 
                                      _compute_shell_disp_qqdd1,_compute_charges,
                                      _compute_coulumb_energy,
                                      _compute_coulumb_energy_pqeq_qd,
                                      _compute_coulumb_energy_pqeq, run_scf)

from data.unpack_tfr_data import unpack_data
from models.ewald import ewald
from models.compute import Compute

import warnings
import logging
constant_e = 1.602176634e-19

#tf.config.run_functions_eagerly(
#    True
#)

#tf.debugging.enable_check_numerics()
class BACENET(tf.keras.Model):
   
    def __init__(self, configs):
        #allows to use all the base class of tf.keras Model
        super().__init__()
        
        #  This part needs to be cleaned up
        self._training_state = None  # mutable mapping
        self.is_training = configs['is_training']
        self.starting_step = configs['initial_global_step']
         
        self.species_layer_sizes = configs['species_layer_sizes']
        layer_sizes = configs['layer_sizes']
        _activations = configs['activations']
        _radial_layer_sizes = configs['radial_layer_sizes']
        #features parameters
        #network section
        self.rcut = configs['rc_rad']
        self.Nrad = int(configs['Nrad'])
        self.n_bessels = configs['n_bessels'] if not None else self.Nrad
        self.n_bessels = int(self.n_bessels)

        #self.zeta = configs['zeta'] if type(configs['zeta'])==list \
        #        else list(range(0,int(configs['zeta'])+1))
        self.zeta = configs['zeta'] # this is a list of l max per body order
        self.nzeta = 0
        for z in self.zeta:
            self.nzeta += z
        #self.nzeta = self.zeta + 1
        #print(f'we are sampling zeta as: {self.zeta} with count {self.nzeta}')
        self.body_order = configs['body_order']
        base_size = self.Nrad * (self.zeta[0] + 1)
        self.feature_size = self.Nrad + base_size
        if self.body_order >= 4 and self.body_order < 8:
            #self.feature_size += self.Nrad * (2*self.nzeta * (self.nzeta+1)) #4 * nzeta * (nzeta+1)/2
            i = 1
            for b in range(4, self.body_order+1):
                self.feature_size += self.Nrad * (1 + self.zeta[i])**(b - 2)
                i += 1
        self.species_correlation = configs['species_correlation']
        self.species_identity = configs['species_identity'] # atomic number
        self.nspecies = len(self.species_identity)
        self.species = configs['species']

        
        self.nspec_embedding = self.species_layer_sizes[-1]
        #if self.species_correlation == 'tensor':
        #    self.spec_size = self.nspec_embedding*self.nspec_embedding
        #else:
        self.spec_size = self.nspec_embedding
        self.feature_size *= self.spec_size
                
        logging.info(f'input dimension for the network features is {self.feature_size}')
        
        #optimization parameters
        self.batch_size = configs['batch_size']
        self.fcost = float(configs['fcost'])
        self.ecost = float(configs['ecost'])
        #if self._train_counter >= configs['start_swa'] and configs['start_swa'] > -1:
        self.fcost_swa = float(configs['fcost_swa'])
        self.ecost_swa = float(configs['ecost_swa'])
        self._start_swa = configs['start_swa_global_step']
        #logging.info(f'ecost will be increased to {self.ecost_swa} when the global time step is > {self._start_swa}')

        #with tf.device('/GPU:0'):
        self.train_writer = tf.summary.create_file_writer(configs['outdir']+'/train')
        self.l1 = float(configs['l1_norm'])
        self.l2 = float(configs['l2_norm'])
        self.learn_radial = configs['learn_radial']


        #dispersion parameters and electrostatics
        self.include_vdw = configs['include_vdw']
        self.rmin_u = configs['rmin_u']
        self.rmax_u = configs['rmax_u']
        self.rmin_d = configs['rmin_d']
        self.rmax_d = configs['rmax_d']
        # the number of elements in the periodic table
        self.nelement = configs['nelement']
        self.coulumb = configs['coulumb']
        self.efield = configs['efield']
        self._sawtooth_PE = configs['sawtooth_PE']
        self._P_in_cell = configs['P_in_cell']
        print('This is P in cell status',self._P_in_cell)
        if self._P_in_cell:
            print('p in cell is active!!!!!')
        self.accuracy = configs['accuracy']
        self.pbc = configs['pbc']
        self.central_atom_id = configs['central_atom_id']
        self.species_nelectrons = configs['species_nelectrons']

        #self.species_nelectrons =  tf.constant([element(sym).nvalence() for sym in self.species])
        #self.species_nelectrons =  tf.constant([element(sym).atomic_number for sym in self.species])
        #'''
        if self.species_nelectrons is None:
            if configs['nshells'] == 0:
                self.species_nelectrons = [2.0 for symb in self.species]
            elif configs['nshells'] == -1: # Use all electrons in the containing unfilled shells
                self.species_nelectrons =  [help_fn.unfilled_orbitals(symb)
                                                    for symb in self.species] # include electrons from l=lmax and l=lmax-1
            else: #select shells to include.
                self.species_nelectrons =  [help_fn.valence_with_two_shells(symb,  
                                                                                    nshells=configs['nshells']) 
                                                    for symb in self.species] # include electrons from l=lmax and l=lmax-1
        #'''
        self.species_nelectrons = tf.cast(self.species_nelectrons, dtype=tf.float32)
        self._initial_shell_sping_constant = configs['initial_shell_spring_constant'] # If this is true, estimate E_d from Zi
        print(f'Shell charges used are: {self.species_nelectrons}')
        # this should be the zeros of the tayloe expansion
        self.oxidation_states = configs['oxidation_states']
        self.learn_oxidation_states = True
        self.gaussian_width_scale = configs['gaussian_width_scale']
        self._qcost = configs['qcost'] 
        


        if self.oxidation_states is None:
            self.oxidation_states = tf.constant([0.0 for i in self.species], tf.float32)

        self.species_chi0 = configs['species_chi0'] * configs['scale_chi0']
        self.species_J0 = configs['species_J0'] * configs['scale_J0']

        self._max_width = configs['max_width']
        #if self._max_width == -1:
        #    self._max_width == 3.0
        self._linearize_d = configs['linearize_d']
        self._anisotropy = configs['anisotropy']
        self.linear_d_terms = configs['linear_d_terms']
        self._d0 = configs['d0'] #initial d0 to enhance training stability
        self.pqeq = configs['pqeq']

        self.atomic_nets = Networks(self.feature_size, 
                    layer_sizes, _activations, 
                    l1=self.l1, l2=self.l2,
        normalize=configs['normalize']) 
        self._normalize = configs['normalize']

        _radial_layer_sizes = configs['radial_layer_sizes']
        self.number_radial_components = 1
        for i, z in enumerate(self.zeta):
            self.number_radial_components += (i+1) * z + 1 # 1 for 2body, l_3 + 1 for 3body, 2 l_4 + 1 for 4body, 3 l_5 + 1 for 5 body

        _radial_layer_sizes.append(self.number_radial_components * self.Nrad)
        radial_activations = ['silu' for s in _radial_layer_sizes[:-1]]
        radial_activations.append('silu')
        self.radial_funct_net = Networks(self.n_bessels, _radial_layer_sizes,
                                         radial_activations,
                                         #l1=self.l1, l2=self.l2
                                 #bias_initializer='zeros',
                                 prefix='radial-functions')

       # create a species embedding network with 1 hidden layer Nembedding x Nspecies
        #species_activations = ['silu' for x in self.species_layer_sizes[:-1]]
        species_activations = ['tanh' for x in self.species_layer_sizes[:-1]]
        species_activations.append('linear')
        nelements = self.nelement

        if self.coulumb: # and self.learn_oxidation_states:
            nelements += 1
        self.species_nets = Networks(nelements, 
                self.species_layer_sizes, 
                species_activations, prefix='species_encoder')

        b_order = self.body_order
        if self.body_order == 40:
            b_order = 4
        self.learnable_gaussian_width = configs['learnable_gaussian_width']
        if self.learnable_gaussian_width:
            #acts = ['softplus', 'softplus']
            #if self._max_width > 0:
            acts = ['tanh', 'tanh']
            self.gaussian_width_net = Networks(self.nelement, # pass one-hot encoder
                [64,2], # return two width per species, one for q and the other for Z 
                acts, 
                prefix='species_gaussian_width')
        
        '''
        self._learn_species_nelectrons = configs['learn_species_nelectrons']
        if self._learn_species_nelectrons:
            self.species_nelectrons_net = Networks(self.nelement, # pass one-hot encoder
                [64,1],
                ['softplus', 'softplus'], prefix='species_nelectrons') # freely learn
        '''

        self.lxlylz = []
        self.lxlylz_sum = []
        self.fact_norm = []
        for i, z in enumerate(self.zeta):
            lxlylz, lxlylz_sum, fact_norm = help_fn._compute_cosine_terms(z)
            lxlylz = tf.cast(lxlylz,tf.float32) #[n_lxlylz, 3]
            lxlylz_sum = tf.cast(lxlylz_sum, tf.int32) #[n_lxlylz,]
            fact_norm = tf.cast(fact_norm, tf.float32) #[n_lxlylz,]
            self.lxlylz.append(lxlylz)
            self.lxlylz_sum.append(lxlylz_sum)
            self.fact_norm.append(fact_norm)

        configs['feature_size'] = self.feature_size 
        configs['spec_size'] = self.spec_size 
        configs['atomic_nets'] = self.atomic_nets 
        configs['radial_funct_net'] = self.radial_funct_net 
        self.model = Compute(configs) # call model

    @tf.function
    def step_fn(self, x):
        x = tf.cast(x, tf.float32)
        return 0.5*(1.0-tf.sign(x))

    @tf.function(jit_compile=False,
                input_signature=[(
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                )])
    def map_fn_parallel_ld0(self, elements):
        out_signature = [
            tf.float32,  # energies
            tf.float32,  # forces
#            tf.float32,  # atomic_features
            tf.float32,  # C6
            tf.float32,  # charges
            tf.float32,  # stress
            tf.float32,  # shell_disp
            tf.float32,  # P
            tf.float32,  # E1
            tf.float32,  # E2
            tf.float32,  # E_d1
            tf.float32,  # E_d2
            tf.float32,  # E_dq
            tf.float32,  # Vj
        ]


        return tf.map_fn(self.model.tf_predict_energy_forces_pqeq0, elements,
                                     fn_output_signature=out_signature,
                                     parallel_iterations=self.batch_size)


    @tf.function(jit_compile=False,
                input_signature=[(
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                )])
    def map_fn_parallel_ld1(self, elements):
        out_signature = [
            tf.float32,  # energies
            tf.float32,  # forces
#            tf.float32,  # atomic_features
            tf.float32,  # C6
            tf.float32,  # charges
            tf.float32,  # stress
            tf.float32,  # shell_disp
            tf.float32,  # P
            tf.float32,  # E1
            tf.float32,  # E2
            tf.float32,  # E_d1
            tf.float32,  # E_d2
            tf.float32,  # E_dq
            tf.float32,  # Vj
        ]


        return tf.map_fn(self.model.tf_predict_energy_forces_pqeq1, elements,
                                     fn_output_signature=out_signature,
                                     parallel_iterations=self.batch_size)


    @tf.function(jit_compile=False,
                input_signature=[(
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,), dtype=tf.int32),
                )])
    def map_fn_parallel(self, elements):
        out_signature = [
            tf.float32,  # energies
            tf.float32,  # forces
#            tf.float32,  # atomic_features
            tf.float32,  # C6
            tf.float32,  # charges
            tf.float32,  # stress
            tf.float32,  # shell_disp
            tf.float32,  # P
            tf.float32,  # E1
            tf.float32,  # E2
            tf.float32,  # Vj
        ]


        return tf.map_fn(self.model.tf_predict_energy_forces_qeq, elements,
                                     fn_output_signature=out_signature,
                                     parallel_iterations=self.batch_size)

    def call(self, data, training=False):
        # may be just call the energy prediction here which will be needed in the train and test steps
        # input contains the following information. Data unpacking are done at the train/test/predict function
        #[0-positions,1-species_encoder,2-C6,3-cells,4-natoms,5-i,6-j,7-S,8-neigh,9-gaussian_width]
        #inputs = unpack_data(inputs)
       
        batch_size = tf.shape(data['positions'])[0] 
        # the batch size may be different from the set batchsize saved in varaible self.batch_size
        # because the number of data point may not be exactly divisible by the self.batch_size.

        batch_nats = tf.cast(tf.reshape(data['natoms'], [-1]), tf.int32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        batch_nmax = tf.tile([nmax], [batch_size])
        nmax_diff = tf.reshape(batch_nmax - batch_nats, [-1])

        #positions and species_encoder are padded tensors.
        positions = tf.reshape(data['positions'][:,:nmax*3], (-1, nmax*3))

        spec_identity = tf.constant(self.species_identity, dtype=tf.int32) # atomic number
        _species_one_hot_encoder = tf.one_hot(spec_identity-1, depth=self.nelement)
        #include the oxidation states as part of species descriptors for long range electrostatic interactions
        if self.coulumb: # and self.learn_oxidation_states:
            species_one_hot_encoder = tf.concat([_species_one_hot_encoder, 
                                             tf.cast(self.oxidation_states,tf.float32)[:,None]], 1)
        else:
            species_one_hot_encoder = _species_one_hot_encoder
        self.trainable_species_encoder = self.species_nets(species_one_hot_encoder) # nspecies x nembedding

        
        #species_encoder = inputs[1].to_tensor(shape=(-1, nmax)) #contains atomic number per atoms for all element in a batch
        atomic_number = tf.reshape(data['atomic_number'][:,:nmax], (-1,nmax)) #contains atomic number per atoms for all element in a batch
        atomic_number = tf.cast(atomic_number, tf.int32)

        # Compare each element in species_encoder to atomic_numbers in spec_identity: It has one true value per row
        matches = tf.equal(tf.cast(atomic_number, tf.float32)[..., tf.newaxis], tf.cast(spec_identity,dtype=tf.float32))  # [batch_size, nmaxx, nspecies]

        # Convert boolean match to one-hot-like index mask and extract all true index including the paddings
        species_indices = tf.argmax(tf.cast(matches, tf.int32), axis=-1)  # [batch_size, nmaxx]

        #create a mask to check for "valid" atomic numbers
        valid_mask = tf.reduce_any(matches, axis=-1)  # [batch_size, nmaxx]
        # Now gather the embeddings
        # self.trainable_species_encoder: shape [S, D]
        batch_species_encoder = tf.gather(self.trainable_species_encoder, species_indices)  # [batch_size, nmaxx, nembedding]
        shape = tf.shape(batch_species_encoder)
        batch_species_encoder = tf.where(valid_mask[..., tf.newaxis],
                                         batch_species_encoder,
                                         tf.zeros(shape))
        #batch_species_encoder = tf.reshape(batch_species_encoder, [batch_size, nmax * self.nspec_embedding])

        #atomic hardness and electronegativity
        #species_chi0, species_J0 = self.estimate_species_chi0_J0()

        if self.coulumb:
            batch_atomic_chi0 = tf.gather(self.species_chi0, species_indices)
            shape = tf.shape(batch_atomic_chi0)
            batch_atomic_chi0 = tf.where(valid_mask,
                                     batch_atomic_chi0,
                                     tf.zeros(shape))

            batch_atomic_J0 = tf.gather(self.species_J0, species_indices)
            batch_atomic_J0 = tf.where(valid_mask,
                                     batch_atomic_J0,
                                     tf.zeros(shape))

            if self.learnable_gaussian_width:
                #minimum = 0.5 and maximum = 2.1, self._species_gaussian_width in [-1,1]
                self._species_gaussian_width = 1.1 + self.gaussian_width_net(_species_one_hot_encoder) # nspec, 2
                #if self._max_width > 0.0:
                #    self._species_gaussian_width = 0.5  + (self._max_width 
                #                                           - 0.5) * self._species_gaussian_width 
                    # species_gaussian_width is between 0,1. spec_gwidth is between 0.5 and 2.5
                #else:
                #    self._species_gaussian_width += 0.5
                batch_gaussian_width = tf.gather(self._species_gaussian_width, species_indices)
                shape = tf.shape(batch_gaussian_width)
                batch_gaussian_width = tf.where(valid_mask[...,tf.newaxis],
                                    batch_gaussian_width,
                                     tf.zeros(shape)) # nbatch, nspec * 2
            #else:
            #    species_gaussian_width = tf.cast(data['gaussian_width'][:,:nmax], tf.float32) * tf.sqrt(2.0) # this defines alpha^2 = 2 * sigma^2
            #    batch_gaussian_width = tf.cast(species_gaussian_width, tf.float32)
            #    self._species_gaussian_width = tf.unique(species_gaussian_width[0])[0]

            #initial charges
            batch_atomic_q0 = tf.gather(tf.cast(self.oxidation_states,tf.float32), species_indices)
            shape = tf.shape(batch_atomic_q0)
            batch_atomic_q0 = tf.where(valid_mask,
                                     batch_atomic_q0,
                                     tf.zeros(shape))
            ### N electrons per species
            ###

            #if self._learn_species_nelectrons:
            #    #override self.species_nelectrons
            #    self.species_nelectrons = tf.reshape(
            #            self.species_nelectrons_net(_species_one_hot_encoder), 
            #                                         [-1]) + 1.0 # min is 1

            self._species_nelectrons = tf.cast(self.species_nelectrons, dtype=tf.float32)
            batch_species_nelec = tf.gather(self._species_nelectrons, species_indices)
            #use atomic number instead
            #batch_species_nelec = tf.gather(tf.cast(spec_identity, tf.float32), species_indices)
            shape = tf.shape(batch_species_nelec)
            batch_species_nelec = tf.where(valid_mask,
                                     batch_species_nelec,
                                     tf.zeros(shape))

            batch_total_charge = tf.reshape(data['total_charge'], [-1])

            self.batch_oxidation_states = tf.reshape(tf.identity(batch_atomic_q0), [batch_size,nmax]) #to regress against predicted charges
        else:
            batch_atomic_chi0 = tf.zeros((batch_size,nmax), dtype=tf.float32)
            batch_gaussian_width = tf.zeros((batch_size,nmax, 2), dtype=tf.float32)
            batch_atomic_J0 = tf.zeros((batch_size,nmax), dtype=tf.float32)
            batch_atomic_q0 = tf.zeros((batch_size,nmax), dtype=tf.float32)
            batch_species_nelec = tf.zeros((batch_size,nmax), dtype=tf.float32)
            batch_total_charge = tf.zeros(batch_size, dtype=tf.float32)

        batch_species_encoder = tf.reshape(batch_species_encoder, [batch_size, nmax * self.nspec_embedding])
        batch_gaussian_width = tf.reshape(batch_gaussian_width, [batch_size, nmax * 2])  

            

       #[0-positions,1-species_encoder,2-C6,3-cells,4-natoms,5-i,6-j,7-S,8-neigh, 9-energy,10-forces]
        C6 = data['C6']
        cells = tf.reshape(data['cells'], [-1, 9])
        #cells = inputs[3]

        #first_atom_idx = tf.cast(inputs[6].to_tensor(shape=(self.batch_size, -1)), tf.int32)
        num_pairs = tf.cast(tf.reshape(data['nneigh'], [-1]), tf.int32)
        neigh_max = tf.reduce_max(num_pairs)
        first_atom_idx = tf.cast(data['i'], tf.int32)
        second_atom_idx = tf.cast(data['j'], tf.int32)
        #shift_vectors = tf.reshape(tf.cast(inputs[7][:,:neigh_max*3],tf.int32), (-1, neigh_max*3))
        shift_vectors = tf.cast(data['S'][:,:neigh_max*3],tf.int32)

        elements = (batch_species_encoder,positions, 
                    nmax_diff, batch_nats,cells,C6, 
                first_atom_idx,second_atom_idx,shift_vectors,num_pairs, batch_gaussian_width,
                    batch_atomic_chi0,batch_atomic_J0,batch_atomic_q0,batch_total_charge, 
                    batch_species_nelec, atomic_number)

        # energies, forces, atomic_features, C6, charges, stress
        if self.pqeq:
            if self._linearize_d == 0:
                energies, forces, C6, charges, stress, shell_disp, Pi_a, E1, E2, E_d2, E_d1, E_qd, Vj = self.map_fn_parallel_ld0(elements)
            else:
                energies, forces, C6, charges, stress, shell_disp, Pi_a, E1, E2, E_d2, E_d1, E_qd, Vj = self.map_fn_parallel_ld1(elements)
        else:
            energies, forces, C6, charges, stress, shell_disp, Pi_a, E1, E2, Vj = self.map_fn_parallel(elements)
            E_d1 = tf.zeros_like(positions)
            E_d2 = tf.zeros_like(positions)
            E_qd = tf.zeros_like(positions)

        #'''
        out_signature = [
            tf.float32,  # energies
            tf.float32,  # forces
#            tf.float32,  # atomic_features
            tf.float32,  # C6
            tf.float32,  # charges
            tf.float32,  # stress
            tf.float32,  # shell_disp
            tf.float32,  # Pi_a
            tf.float32,  # E1
            tf.float32,  # E2
            tf.float32,  # E_d1
            tf.float32,  # E_d2
            tf.float32,  # E_dq
            tf.float32,  # Vj

        ]


        #energies, forces, C6, charges, stress, shell_disp = tf.map_fn(self.tf_predict_energy_forces, elements,
        #                             fn_output_signature=out_signature,
        #                             parallel_iterations=self.batch_size)
        #'''
        #outs = [energies, forces, atomic_features]
        outs = {'energy':energies,
                'forces':forces,
                #'features':atomic_features,
                'C6': C6,
                'charges': charges,
                'stress': stress,
                'shell_disp':shell_disp,
                'Pi_a':Pi_a,
                'E1':E1,
                'E2':E2,
                'E_d1':E_d1,
                'E_qd':E_qd,
                'E_d2':E_d2,
                'Vj':Vj,
                }
        return outs

    def compile(self, optimizer, loss, loss_f, loss_q):
        super().compile()
        self.optimizer = optimizer
        self.loss_e = loss
        self.loss_f = loss_f
        self.loss_q = loss_q

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh, energy,forces]
        #target = tf.cast(tf.reshape(inputs_target[9], [-1]), tf.float32)
        batch_nats = tf.cast(tf.reshape(data['natoms'], [-1]), tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        
        #target_f = tf.reshape(inputs_target[7], [-1, 3*nmax])
        #target_f = tf.cast(target_f, tf.float32)
        
        with tf.GradientTape() as tape:
            #outs = self(inputs_target[:10], training=True)
            outs = self(data, training=True)
            e_pred = outs['energy']
            forces = outs['forces']
            stress = outs['stress']
            #Zstar = outs['Zstar']
            #features = outs['features']
            C6 = charges = None
            if self.include_vdw:
                C6 = outs['C6']
            if self.coulumb:
                charges = outs['charges']

            target = tf.reshape(data['energy'], [-1])
            target_f = data['forces'][:,:nmax*3]

            ediff = (e_pred - target)
            forces = tf.reshape(forces, [-1, 3*nmax])
            emse_loss = self.loss_e(target, e_pred)

            fmse_loss = tf.map_fn(self.loss_f, 
                                  (batch_nats,target_f,forces), 
                                  fn_output_signature=tf.float32)
            fmse_loss = tf.reduce_mean(fmse_loss) # average over batch

            delta_n = (tf.constant(self._start_swa, dtype=tf.int64) - 
                       (self._train_counter + self.starting_step))
            _step_fn = self.step_fn(delta_n)
            _ecost = self.ecost + (self.ecost_swa - self.ecost) * _step_fn 
            #=self.ecost if self._start_swa > self._train_counter and self.ecost_swa otherwise
            _fcost = self.fcost + (self.fcost_swa - self.fcost) * _step_fn 
            #=self.fcost if self._start_swa > self._train_counter and self.fcost_swa otherwise

            _loss = _ecost * emse_loss
            _loss += _fcost * fmse_loss
            #if self.learn_oxidation_states:
            #    q_loss = tf.map_fn(self.loss_q,
            #                      (self.batch_oxidation_states,charges,batch_nats),
            #                      fn_output_signature=tf.float32)
            #    q_loss = self._qcost * tf.reduce_mean(q_loss)
            #    _loss += q_loss
            #else:
            q_loss = 0.0
        # Compute gradients
        trainable_vars = self.trainable_variables
        #assert trainable_vars == self.trainable_weights
        #trainable_vars = self.trainable_weights
        
        gradients = tape.gradient(_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        #self._train_counter.assign_add(1)



        metrics = {'tot_st': self._train_counter + self.starting_step}
        #metrics = {}
        mae = tf.reduce_mean(tf.abs(ediff / batch_nats))
        rmse = tf.sqrt(tf.reduce_mean((ediff/batch_nats)**2))

        mae_f = tf.map_fn(help_fn.force_mae, 
                          (batch_nats,target_f,forces), 
                          fn_output_signature=tf.float32)
        mae_f = tf.reduce_mean(mae_f)

        rmse_f = tf.sqrt(fmse_loss)

        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})

        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})
        metrics.update({'loss': _loss})
        metrics.update({'energy loss': _ecost*emse_loss})
        metrics.update({'force loss': _fcost*fmse_loss})
        metrics.update({'charge loss': q_loss})

        #with writer.set_as_default():
        #'''
        with self.train_writer.as_default(step=self._train_counter):
        #with self.train_writer.as_default():

            tf.summary.scalar('1. Losses/1. Total',_loss, self._train_counter + self.starting_step)
            tf.summary.scalar('1. Losses/2. Energy',emse_loss,self._train_counter + self.starting_step)
            tf.summary.scalar('1. Losses/3. Forces',fmse_loss,self._train_counter + self.starting_step)
            tf.summary.scalar('2. Metrics/1. RMSE/atom',rmse,self._train_counter + self.starting_step)
            tf.summary.scalar('2. Metrics/2. MAE/atom',mae,self._train_counter + self.starting_step)
            tf.summary.scalar('2. Metrics/3. RMSE_F',rmse_f,self._train_counter + self.starting_step)
            tf.summary.scalar('2. Metrics/4. MAE_F',mae_f,self._train_counter + self.starting_step)
            #tf.summary.histogram(f'3. angular terms: lambda',self._lambda_weights,self._train_counter)
            for idx, spec in enumerate(self.species_identity):
                tf.summary.histogram(f'3. encoding /{spec}',self.trainable_species_encoder[idx],self._train_counter + self.starting_step)
                tf.summary.scalar(f'31. gaussian_width_q /{spec}',self._species_gaussian_width[idx,0],self._train_counter + self.starting_step)
                tf.summary.scalar(f'31. gaussian_width_Z /{spec}',self._species_gaussian_width[idx,1],self._train_counter + self.starting_step)
                tf.summary.scalar(f'32. shell  charges /{spec}',self._species_nelectrons[idx],self._train_counter + self.starting_step)
            if self.include_vdw:
                tf.summary.histogram(f'4. C6 parameters',C6, self._train_counter + self.starting_step)
            if self.coulumb:
                tf.summary.histogram(f'5. charges',charges, self._train_counter + self.starting_step)

        #'''
        return {key: metrics[key] for key in metrics.keys()}


    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        #inputs_target = unpack_data(data)
        batch_nats = tf.cast(tf.reshape(data['natoms'], [-1]), tf.float32)

        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)

        #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh, energy,forces]
        outs = self(data, training=True)
        #outs = self(data, training=True)
        e_pred = outs['energy']
        forces = outs['forces']
        stress = outs['stress']
        #Zstar = outs['Zstar']
        #features = outs['features']
        #C6 = charges = None
        #if self.include_vdw:
        C6 = outs['C6']
        #if self.coulumb:
        charges = outs['charges']
        target = tf.cast(tf.reshape(data['energy'], [-1]), tf.float32)
        target_f = tf.reshape(data['forces'][:,:nmax*3], [-1, 3*nmax])

        #e_pred, forces, _ = self(inputs_target[:9], training=True)  # Forward pass
        forces = tf.reshape(forces, [-1, nmax*3])
        #target_f = tf.reshape(inputs_target[7], [-1, 3*nmax])
        target_f = tf.cast(target_f, tf.float32)

        ediff = (e_pred - target)

        mae = tf.reduce_mean(tf.abs(ediff / batch_nats))
        rmse = tf.sqrt(tf.reduce_mean((ediff/batch_nats)**2))

        fmse_loss = tf.map_fn(help_fn.force_loss, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        fmse_loss = tf.reduce_mean(fmse_loss)

        mae_f = tf.map_fn(help_fn.force_mae, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        mae_f = tf.reduce_mean(mae_f)
        rmse_f = tf.sqrt(fmse_loss)

        metrics = {}
        loss = self.ecost * rmse * rmse + self.fcost * fmse_loss

        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})
        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})
        metrics.update({'loss': loss})
        with self.train_writer.as_default(step=self._train_counter + self.starting_step):
            tf.summary.scalar('2. Metrics/1. V_RMSE/atom',rmse,self._train_counter + self.starting_step)
            tf.summary.scalar('2. Metrics/2. V_MAE/atom',mae,self._train_counter + self.starting_step)
            tf.summary.scalar('2. Metrics/3. V_RMSE_F',rmse_f,self._train_counter + self.starting_step)
            tf.summary.scalar('2. Metrics/3. V_MAE_F',mae_f,self._train_counter + self.starting_step)



        return {key: metrics[key] for key in metrics.keys()}


    def predict_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        #inputs_target = unpack_data(data)
        batch_nats = tf.cast(tf.reshape(data['natoms'], [-1]), tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
       # target_f = tf.reshape(inputs_target[10][:,:nmax*3], [-1, 3*nmax])
        outs = self(data, training=False)
        e_pred = outs['energy']
        forces = outs['forces']
        target = tf.cast(tf.reshape(data['energy'], [-1]), tf.float32)
        target_f = tf.reshape(data['forces'][:,:nmax*3], [-1, 3*nmax])

#        e_pred, forces, _ = self(inputs_target[:9], training=True)  # Forward pass
        forces = tf.reshape(forces, [-1, nmax*3])
        target_f = tf.cast(target_f, tf.float32)

        ediff = (e_pred - target)

        forces_ref = tf.reshape(target_f, [-1, nmax, 3])
        forces_pred = tf.reshape(forces, [-1, nmax, 3])

        #if self.coulumb:
        charges = tf.reshape(outs['charges'], [-1,nmax])
        return [target, e_pred, forces_ref, forces_pred, 
                batch_nats, charges,outs]
