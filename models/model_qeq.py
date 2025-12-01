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
                                      _compute_coulumb_energy_pqeq )

from data.unpack_tfr_data import unpack_data
from models.ewald import ewald

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
        
        #self._training_state = {}    # allow Keras to populate counters here
        self._training_state = None  # mutable mapping
        self.is_training = configs['is_training']
        self.starting_step = configs['initial_global_step']
         
        self.species_layer_sizes = configs['species_layer_sizes']
        layer_sizes = configs['layer_sizes']
        _activations = configs['activations']
        _radial_layer_sizes = configs['radial_layer_sizes']
        #features parameters

        #self.loss_tracker = self.metrics.Mean(name='loss')
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
        if self.body_order == 40:
            self.feature_size += self.Nrad * (self.zeta[1] + 1)**3
            #self.feature_size += self.Nrad
        self.species_correlation = configs['species_correlation']
        self.species_identity = configs['species_identity'] # atomic number
        self.nspecies = len(self.species_identity)
        self.species = configs['species']

        
        self.nspec_embedding = self.species_layer_sizes[-1]
        if self.species_correlation == 'tensor':
            self.spec_size = self.nspec_embedding*self.nspec_embedding
        else:
            self.spec_size = self.nspec_embedding
        self.feature_size *= self.spec_size
        self.feature_size += 1 # dot(efield, positions), for the electric field coupling with atomic position. this is just concatenated to the descriptors for now
                
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
        if self.efield is not None:
            self.efield = tf.cast(self.efield, tf.float32)
            self.apply_field = True
        else:
            self.efield = tf.cast([0.0,0.0,0.0], tf.float32)
            self.apply_field = False
        print('field', self.efield)

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
        self.species_nelectrons = tf.cast(self.species_nelectrons, dtype=tf.float32)
        self._initial_shell_sping_constant = configs['initial_shell_spring_constant'] # If this is true, estimate E_d from Zi
        #print(f'Shell charges used are: {self.species_nelectrons}')
        # this should be the zeros of the tayloe expansion
        self.oxidation_states = configs['oxidation_states']
        self.learn_oxidation_states = True
        self.gaussian_width_scale = configs['gaussian_width_scale']
        self._qcost = configs['qcost'] 

        if self.oxidation_states is None:
            self.oxidation_states = tf.constant([0.0 for i in self.species], tf.float32)
        # Need a few checks to make sure that q+Z or q-Z is finite. Adhoc correction, add 1 to Z
        #for i in range(len(self.species)):
        #    if self.oxidation_states[i] == self.species_nelectrons[i]
        #    self.species_nelectrons += 1 #adhoc

        mse_os = 0.0 # this is not a tensor object
        for i, oxs in enumerate(self.oxidation_states):
            mse_os += oxs * oxs

        if mse_os < 1e-3  or self._qcost == 0.0:
            self.learn_oxidation_states = False
        #else:
        #    print(f'Learning of oxidation state: {self.oxidation_states}')

        self.species_chi0 = configs['species_chi0'] * configs['scale_chi0']
        self.species_J0 = configs['species_J0'] * configs['scale_J0']

        self._max_width = configs['max_width']
        #if self._max_width == -1:
        #    self._max_width == 3.0
        self._linearize_d = configs['linearize_d']
        self._anisotropy = configs['anisotropy']
        self.linear_d_terms = configs['linear_d_terms']
        self._d0 = configs['d0'] #initial d0 to enhance training stability
        if self.linear_d_terms and self._anisotropy:
            if layer_sizes[-1] != 12:
                raise ValueError("The last layer must be 12")
        elif self.linear_d_terms and not self._anisotropy:
            if layer_sizes[-1] != 6:
                raise ValueError("The last layer must be 12")
        if self._anisotropy:

            if layer_sizes[-1] != 12:
                raise ValueError("The last layer must be 12")



        #if not self.species_layer_sizes:
        #    self.species_layer_sizes = [self.nspec_embedding]
        self.features = configs['features']
        #if self.coulumb:
        #    self.feature_size += 1

        self.pqeq = configs['pqeq']

        self.atomic_nets = Networks(self.feature_size, 
                    layer_sizes, _activations, 
                    l1=self.l1, l2=self.l2,
        normalize=configs['normalize']) 
        self._normalize = configs['normalize']

        #self.atomic_nets = atomic_nets
        #self.species_nets = species_nets
        #self.radial_funct_net = radial_funct_net

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

        #radial network
        # each body order learn single component of the radial functions
        #one radial function per components and different one for 3 and 4 body 
        #if self.body_order == 3:
        #self.number_radial_components = self.nzeta # this is lmax + 1
        #if self.body_order == 4:
        #    self.number_radial_components = 2 * self.nzeta - 1
        b_order = self.body_order
        if self.body_order == 40:
            b_order = 4
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
        self.learnable_gaussian_width = configs['learnable_gaussian_width']
        if self.learnable_gaussian_width:
            #acts = ['softplus', 'softplus']
            #if self._max_width > 0:
            acts = ['sigmoid', 'sigmoid']
            self.gaussian_width_net = Networks(self.nelement, # pass one-hot encoder
                [64,2], # return two width per species, one for q and the other for Z 
                acts, 
                prefix='species_gaussian_width')

        self._learn_species_nelectrons = configs['learn_species_nelectrons']
        if self._learn_species_nelectrons:
            self.species_nelectrons_net = Networks(self.nelement, # pass one-hot encoder
                [64,1],
                ['softplus', 'softplus'], prefix='species_nelectrons') # freely learn
            #inputs = tf.one_hot([81,21,7],depth=128)
            #print(tf.reshape(self.species_nelectrons_net(inputs), [-1]), self.species_nelectrons)
            #print(tf.reshape(self.species_nelectrons_net(inputs), [-1]) + self.species_nelectrons)

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

        #self.lxlylz = tf.convert_to_tensor(lxlylz, dtype=tf.float32)
        #self.lxlylz_sum = tf.convert_to_tensor(lxlylz_sum, tf.int32)
        #self.fact_norm = tf.convert_to_te
        #self.species_gaussian_width = self.nsor(fact_norm, dtype=tf.float32)
        #initializer for lambda

    @tf.function
    def step_fn(self, x):
        x = tf.cast(x, tf.float32)
        return 0.5*(1.0-tf.sign(x))

    @tf.function(jit_compile=False,
                 reduce_retracing=True,
                input_signature=[(
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                )])
    def tf_predict_energy_forces(self,x):
        ''' 
        x = (batch_species_encoder,positions,
                    nmax_diff, batch_nats,cells,C6,
                first_atom_idx,second_atom_idx,shift_vectors,num_pairs)
        '''
        rc = tf.constant(self.rcut,dtype=tf.float32)
        Nrad = tf.constant(self.Nrad, dtype=tf.int32)
        _efield = self.efield
        apply_field = self.apply_field

        #thetasN = tf.constant(self.thetaN, dtype=tf.int32)

        nat = x[3]
        nmax_diff = x[2]
        
        species_encoder = tf.reshape(x[0][:nat*self.nspec_embedding], 
                                     [nat,self.nspec_embedding])

        #width = tf.cast(x[1], dtype=tf.float32)
        positions = tf.reshape(x[1][:nat*3], [nat,3])
        positions = positions
        cell = tf.reshape(x[4], [3,3])
        evdw = 0.0
        C6 = x[5][:nat]
        num_pairs = x[9]
        first_atom_idx = tf.cast(x[6][:num_pairs], tf.int32)
        second_atom_idx = tf.cast(x[7][:num_pairs],tf.int32)
        shift_vector = tf.cast(tf.reshape(x[8][:num_pairs*3], 
                                          [num_pairs,3]), tf.float32)
        gaussian_width = tf.reshape(x[10][:nat*2], [nat,2])
        chi0 = x[11][:nat]
        J0 = x[12][:nat]
        atomic_q0 = x[13][:nat]
        total_charge = x[14]
        nuclei_charge = x[15][:nat]
        atomic_number = x[16][:nat]

        with tf.GradientTape(persistent=True) as tape0:
            #'''
            tape0.watch(positions)
            tape0.watch(cell)
            tape0.watch(_efield)

            #based on ase 
            #npairs x 3
            all_rij = tf.gather(positions,second_atom_idx) - \
                     tf.gather(positions,first_atom_idx) + \
                    tf.tensordot(shift_vector,cell,axes=1)

            all_rij_norm = tf.linalg.norm(all_rij, axis=-1) #npair
            reg = 1e-12
            #all_rij_norm = tf.sqrt(tf.reduce_sum(all_rij * all_rij , axis=-1) + reg) #npair
            species_encoder_i = tf.gather(species_encoder,first_atom_idx)
            species_encoder_j = tf.gather(species_encoder,second_atom_idx)
            if self.species_correlation=='tensor':
                species_encoder_extended = tf.expand_dims(species_encoder_i, -1) * tf.expand_dims(species_encoder_j, -2)

            else:
                species_encoder_extended = species_encoder_i * species_encoder_j

            species_encoder_ij = tf.reshape(species_encoder_extended, 
                                                  [-1, self.spec_size]
                                                   ) #npairs,spec_size

            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x Nneigh x embedding
            #fcuts is nat x Nneigh and reshaped to nat x Nneigh
            #_Nneigh = tf.shape(all_rij_norm)
            #neigh = _Nneigh[1]
            kn_rad = tf.ones(self.n_bessels,dtype=tf.float32)
            bf_radial0 = help_fn.bessel_function(all_rij_norm,
                                            rc,kn_rad,
                                            self.n_bessels) #

            bf_radial1 = tf.reshape(bf_radial0, [-1,self.n_bessels])
            bf_radial2 = self.radial_funct_net(bf_radial1)
            bf_radial = tf.reshape(bf_radial2, [num_pairs, self.Nrad, self.number_radial_components])
            radial_ij = tf.expand_dims(bf_radial, 2) * tf.expand_dims(tf.expand_dims(species_encoder_ij, 1), -1)
            radial_ij = tf.reshape(radial_ij, [num_pairs, self.Nrad*self.spec_size,self.number_radial_components])
            atomic_descriptors = tf.math.unsorted_segment_sum(data=radial_ij[:,:,0],
                                                              segment_ids=first_atom_idx, num_segments=nat) 

            #implement angular part: compute vect_rij dot vect_rik / rij / rik
            #rij_unit = tf.einsum('ij,i->ij',all_rij, 1.0 / (all_rij_norm+reg)) #npair,3
            rij_unit = all_rij / (tf.expand_dims(all_rij_norm + reg, -1))

            Gi = to_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat, self.body_order)
            for nbody in tf.range(3, self.body_order+1):
                atomic_descriptors = tf.concat([atomic_descriptors, Gi[nbody-3]], axis=1)

            #field components = \sum_j{fc(rij) * (E . rij_unit)}
            _efield_extended = tf.squeeze(tf.matmul(rij_unit, _efield[:, None])) * help_fn.tf_fcut_rbf(all_rij_norm, rc)
            _efield_extended = tf.math.unsorted_segment_sum(data=_efield_extended,
                                                              segment_ids=first_atom_idx, num_segments=nat)

            atomic_descriptors = tf.concat([atomic_descriptors, _efield_extended[:,None]], axis=1) # append electric field
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])

            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            total_energy = tf.reduce_sum(_atomic_energies[:,0])
            idx = 1
            E1 = tf.nn.softplus(_atomic_energies[:,idx])
            idx += 1
            E2 = tf.nn.softplus(_atomic_energies[:,idx])
            #include atomic electronegativity(chi0) and hardness (J0)
            E1 += chi0
            E2 += J0

            _b = tf.identity(E1)
            _b -= E2 * atomic_q0 # only if we are not optimizing deq

            _ewald = ewald(positions, cell, nat,
                    gaussian_width,self.accuracy, 
                           None, self.pbc, _efield,
                           self.gaussian_width_scale 
                           )
            Vij = _ewald.recip_space_term() if self.pbc else _ewald.real_space_term()
            if apply_field:
                field_kernel, field_kernel_e = _ewald.potential_linearized_periodic_ref0(tf.zeros_like(nuclei_charge))
                _b += field_kernel
            charges = self.compute_charges(Vij, _b, E2, atomic_q0, total_charge)
            ecoul = self.compute_coulumb_energy(charges, atomic_q0, E1, E2, Vij)
            Piq_a, Pie_a = _ewald.atom_centered_polarization(tf.zeros_like(positions),
                                                             tf.zeros_like(charges),
                                                             charges,
                                                             self.central_atom_id,
                                                             atomic_number
                                                             )
            #Pi_a = tf.stack([Piq_a + Pie_a, Piq_a, Pie_a])
            Pi_a = Piq_a + Pie_a
            if apply_field:
                efield_energy = tf.reduce_sum(charges * field_kernel)
                ecoul += efield_energy

            total_energy += ecoul
            Vj = _ewald.recip_space_term_with_shelld_linear_Vj(shell_disp,
                                               nuclei_charge,
                                               charges)

        #differentiating a scalar w.r.t tensors
        pad_rows = tf.zeros([nmax_diff], dtype=tf.float32)
        charges = tf.concat([charges, pad_rows], axis=0)
        Pi_a = -tape0.gradient(total_energy, _efield) #contains P0 - dE/d_eps
        #total_energy -= tf.reduce_sum(Pi_a * _efield)
        forces = tape0.gradient(total_energy, positions)
        #needs tape to be persistent
        if self.pbc:
            dE_dh = tape0.gradient(total_energy, cell)
            #V = tf.abs(tf.linalg.det(cell))
            V = tf.abs(tf.tensordot(cell[0], tf.linalg.cross(cell[1], cell[2]),axes=1))
            # Stress: stress_{ij} = (1/V) \sum_k dE/dh_{ik} * h_{jk}
            stress = tf.linalg.matmul(dE_dh, cell, transpose_b=True) / V
        else:
            stress = tf.zeros((3,3), dtype=tf.float32)
        pad_rows = tf.zeros([nmax_diff, tf.shape(forces)[1]], dtype=forces.dtype)
        forces = tf.concat([-forces, pad_rows], axis=0)
        shell_disp = tf.concat([shell_disp, pad_rows], axis=0)

        #forces = tf.pad(-forces, paddings=[[0,nmax_diff],[0,0]], constant_values=0.0)
        return [total_energy, forces, C6, charges, stress, shell_disp, Pi_a, E1, E2, E_d2,E_d1,E_qd, Vj]

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
            tf.float32,  # E_d1
            tf.float32,  # E_d2
            tf.float32,  # E_dq
            tf.float32,  # Vj
        ]


        return tf.map_fn(self.tf_predict_energy_forces, elements,
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
                #minimum = 0.2 and maximum = 1.2, self._species_gaussian_width in [0,1]
                self._species_gaussian_width = 0.25 + self.gaussian_width_net(_species_one_hot_encoder) # nspec, 2
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
            if self._learn_species_nelectrons:
                #override self.species_nelectrons
                self.species_nelectrons = tf.reshape(
                        self.species_nelectrons_net(_species_one_hot_encoder), 
                                                     [-1]) + 1.0 # min is 1

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
        energies, forces, C6, charges, stress, shell_disp, Pi_a, E1, E2, E_d2, E_d1, E_qd, Vj = self.map_fn_parallel(elements)
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
            if self.learn_oxidation_states:
                q_loss = tf.map_fn(self.loss_q,
                                  (self.batch_oxidation_states,charges,batch_nats),
                                  fn_output_signature=tf.float32)
                q_loss = self._qcost * tf.reduce_mean(q_loss)
                _loss += q_loss
            else:
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
