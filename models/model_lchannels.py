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

from data.unpack_tfr_data import unpack_data
from functions.ewald import ewald

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
        self.zeta = configs['zeta'] # this now an integer determining the max l 
        self.nzeta = self.zeta + 1
        self.n_lambda = configs['n_lambda']
        #print(f'we are sampling zeta as: {self.zeta} with count {self.nzeta}')
        self.body_order = configs['body_order']
        base_size = self.Nrad * (self.nzeta)
        self.feature_size = self.Nrad + base_size
        if self.body_order == 4:
            #self.feature_size += self.Nrad * (2*self.nzeta * (self.nzeta+1)) #4 * nzeta * (nzeta+1)/2
            self.feature_size += self.Nrad * self.nzeta**2
            #self.feature_size += self.Nrad
        self.species_correlation = configs['species_correlation']
        self.species_identity = configs['species_identity'] # atomic number
        self.nspecies = len(self.species_identity)
        self.species = configs['species']
        # this should be the zeros of the tayloe expansion
        self.oxidation_states = configs['oxidation_states']
        if self.oxidation_states is None:
            self.oxidation_states = [0.0 for i in self.species]

        self.species_chi0 = configs['species_chi0']
        self.species_J0 = configs['species_J0']

        self.nspec_embedding = self.species_layer_sizes[-1]
        if self.species_correlation == 'tensor':
            self.spec_size = self.nspec_embedding*self.nspec_embedding
        else:
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
        print(f'ecost will be increased to {self.ecost_swa} when the global time step is > {self._start_swa}')

        print(self.fcost, self.ecost, self._train_counter)
        #with tf.device('/GPU:0'):
        self.train_writer = tf.summary.create_file_writer(configs['outdir']+'/train')
        self.l1 = float(configs['l1_norm'])
        self.l2 = float(configs['l2_norm'])
        self.learn_radial = configs['learn_radial']


        #dispersion parameters
        self.include_vdw = configs['include_vdw']
        self.rmin_u = configs['rmin_u']
        self.rmax_u = configs['rmax_u']
        self.rmin_d = configs['rmin_d']
        self.rmax_d = configs['rmax_d']
        # the number of elements in the periodic table
        self.nelement = configs['nelement']
        self.coulumb = configs['coulumb']
        self.efield = configs['efield']
        self.accuracy = configs['accuracy']
        self.pbc = configs['pbc']
        #self.total_charge = configs['total_charge']
        
        #if not self.species_layer_sizes:
        #    self.species_layer_sizes = [self.nspec_embedding]
        self.features = configs['features']
        if self.coulumb:
            self.feature_size += 1
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

        self.species_nets = Networks(self.nelement, 
                self.species_layer_sizes, 
                species_activations, prefix='species_encoder')

        #radial network
        # each body order learn single component of the radial functions
        #one radial function per components and different one for 3 and 4 body 
        #if self.body_order == 3:
        self.number_radial_components = self.nzeta # this is lmax + 1
        if self.body_order == 4:
            self.number_radial_components = 2 * self.nzeta - 1

        _radial_layer_sizes.append(self.number_radial_components * self.Nrad)
        radial_activations = ['silu' for s in _radial_layer_sizes[:-1]]
        radial_activations.append('silu')
        self.radial_funct_net = Networks(self.n_bessels, _radial_layer_sizes, 
                                         radial_activations, 
                                         #l1=self.l1, l2=self.l2
                                 #bias_initializer='zeros',
                                 prefix='radial-functions')
        #lxlylz, lxlylz_sum, fact_norm = help_fn.precompute_fact_norm_lxlylz(self.zeta)
        lxlylz, lxlylz_sum, fact_norm = help_fn._compute_cosine_terms(self.zeta)
        self.lxlylz = tf.cast(lxlylz,tf.float32) #[n_lxlylz, 3]
        self.lxlylz_sum = tf.cast(lxlylz_sum, tf.int32) #[n_lxlylz,]
        self.fact_norm = tf.cast(fact_norm, tf.float32) #[n_lxlylz,]

        #self.lxlylz = tf.convert_to_tensor(lxlylz, dtype=tf.float32)
        #self.lxlylz_sum = tf.convert_to_tensor(lxlylz_sum, tf.int32)
        #self.fact_norm = tf.convert_to_tensor(fact_norm, dtype=tf.float32)
        #initializer for lambda

    @tf.function
    def step_fn(self, x):
        x = tf.cast(x, tf.float32)
        return 0.5*(1.0-tf.sign(x))

    @tf.function(jit_compile=True)
    def custom_segment_sum(self, data, segment_ids, num_segments):
        result_shape = tf.concat([[num_segments], tf.shape(data)[1:]], axis=0)
        zeros = tf.zeros(result_shape, dtype=data.dtype)
        indices = tf.expand_dims(segment_ids, axis=-1)  # shape (N, 1)
        return tf.tensor_scatter_nd_add(zeros, indices, data)

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def _angular_terms(self, rij_unit):
        '''
        Compute vectorized three-body angular terms.
        '''
        # Compute powers: shape = [npairs, n_lxlylz, 3]
        rij_lxlylz = (tf.expand_dims(rij_unit, axis=1) + 1e-12) ** tf.expand_dims(self.lxlylz, axis=0)

        # Multiply x^lx * y^ly * z^lz
        #g_ij_lxlylz = tf.reduce_prod(rij_lxlylz, axis=-1)              # [npairs, n_lxlylz]
        g_ij_lxlylz = rij_lxlylz[:,:,0] * rij_lxlylz[:,:,1] * rij_lxlylz[:,:,2]  

        # Apply factorial norm (broadcasted)
        #fact_norm = tf.reshape(fact_norm, [-1])                        # [n_lxlylz]
        #g_ij_lxlylz = g_ij_lxlylz * self.fact_norm[None,:]                # [npairs, n_lxlylz]

        return g_ij_lxlylz

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                 )
    def _to_three_body_terms(self, z, rij_unit, radial_ij, first_atom_idx, nat):
        """
        Computes vectorized three-body symmetry function components.

        Args:
            z: scalar (int) orbital angular momentum order
            rij_unit: Tensor, shape = [npair, 3] — unit vectors between atom pairs
            radial_ij: Tensor, shape = [npair, nspec * nrad, nzeta] — radial features
            first_atom_idx: Tensor, shape = [npair] — maps each neighbor pair to a central atom index
            lambda_weights: Tensor, shape = [n_lambda] — weights for lambda values
            nat: scalar (int), number of atoms in batch

        Returns:
            Tensor of shape [nat, nzeta * nspec * nrad * n_lambda]
        """
         # --- Angular terms ---
        g_ij_lxlylz = self._angular_terms(rij_unit)
        # shape: [npair, n_lxlylz]
        #radial_ij -> shape: [npair, nspec * nrad, nzeta] #0 to zeta

        #radial_ij = tf.transpose(radial_ij, perm=[2,0,1])
        #radial_ij_expanded = tf.transpose(tf.gather(radial_ij, self.lxlylz_sum), perm=[1,2,0])
        #radial_ij_expanded = tf.gather(radial_ij, self.lxlylz_sum, axis=2)
        radial_ij_expanded = radial_ij
        # shape: [npair, nspec * nrad, n_lxlylz]

        g_ilxlylz = radial_ij_expanded * tf.expand_dims(g_ij_lxlylz, axis=1)
        #sum over neighbors
        g_ilxlylz = tf.math.unsorted_segment_sum(data=g_ilxlylz,
                                            segment_ids=first_atom_idx, num_segments=nat) #shape = (nat, nrad*nspec, nzeta)
        #sum over z
        #lambda_comp = tf.constant([1.0,-1.0])
        #lambda_comp = lambda_weights
        #lambda_comp_lxlylz = lambda_comp[:,None]**tf.cast(lxlylz_sum,tf.float32)[None,:] #n_lambda_comp, nlxlylz
        #gi3 = tf.einsum('ijk,lk->ijl', g2_ilxlylz, lambda_comp_lxlylz)
        #there is no need for different lambdas

        # Multiply g_ilxlylz^2 and transpose
        _gi3 = tf.transpose(g_ilxlylz * g_ilxlylz, [2,0,1]) * self.fact_norm[:,None,None] #n_lxlylz, nat, nspec * nrad
        #_gi3 = tf.einsum('ijk,ijk->kij', g_ilxlylz, g_ilxlylz)

        gi3 = tf.math.unsorted_segment_sum(_gi3, self.lxlylz_sum, num_segments=self.nzeta)
        #gi3 = self.custom_segment_sum(_gi3, self.lxlylz_sum, num_segments=self.nzeta)
        # shape: [nzeta, nat, nspec * nrad]

        gi3 = tf.transpose(gi3, perm=(1,0,2)) #nat,nzeta,nrad*nspec
        # shape: [nat,nzeta,,nspec * nrad, n_lambda]
        #norm = tf.pow(2.0, 1.0 - tf.cast(z, tf.float32))
        #gi3 *= norm

        return tf.reshape(gi3, [nat, -1])
        # final shape: [nat, nzeta * nspec * nrad]

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                 )
    def _to_four_body_terms(self,z, rij_unit, radial_ij, first_atom_idx, nat):
        '''
        compute  up to four-body computation

        '''
        # --- Angular terms ---
        g_ij_lxlylz = self._angular_terms(rij_unit)
        shapes = tf.shape(g_ij_lxlylz)
        npairs = shapes[0]
        n_lxlylz = shapes[1]
        # shape: [npair, n_lxlylz]
        #radial_ij = tf.transpose(radial_ij, perm=[2,0,1])
        #radial_ij_expanded = tf.transpose(tf.gather(radial_ij, self.lxlylz_sum), perm=[1,2,0])
        #radial_ij_expanded = tf.gather(radial_ij, self.lxlylz_sum, axis=2)
        radial_ij_expanded = tf.gather(radial_ij[:,:,:self.nzeta], self.lxlylz_sum, axis=2)
        #radial_ij_expanded = radial_ij
        # shape: [npair, nspec * nrad, n_lxlylz]

        g_ilxlylz = radial_ij_expanded * tf.expand_dims(g_ij_lxlylz, axis=1)
        # shape: [npair, nspec * nrad, n_lxlylz]

        g_ilxlylz = tf.math.unsorted_segment_sum(g_ilxlylz, first_atom_idx,num_segments=nat)
        #g_ilxlylz = self.custom_segment_sum(g_ilxlylz, first_atom_idx,nat)
        # shape: [nat, nspec * nrad, n_lxlylz]

        #lambda_comp_lxlylz = lambda_weights[:, None] ** tf.cast(lxlylz_sum, tf.float32)[None, :]
        #lambda_comp_n = lambda_weights[None,:] ** tf.range(self.nzeta,dtype=tf.float32)[:, None]
        # shape: [nzeta,n_lambda]

        # Multiply g_ilxlylz^2 and transpose
        _gi3 = tf.transpose(g_ilxlylz * g_ilxlylz, [2,0,1]) * self.fact_norm[:,None,None]
        
        #_gi3 = tf.einsum('ijk,ijk->kij', g_ilxlylz, g_ilxlylz)
        # shape: [n_lxlylz, nat, nspec * nrad]

        gi3 = tf.math.unsorted_segment_sum(_gi3, self.lxlylz_sum, num_segments=self.nzeta)
        #gi3 = self.custom_segment_sum(_gi3, lxlylz_sum, num_segments=self.nzeta)
        # shape: [nzeta, nat, nspec * nrad]

        #gi3 = tf.einsum('ijk, il->jikl', gi3, lambda_comp_n)
        # shape: [nat,nzeta,,nspec * nrad, n_lambda]
        #norm = tf.pow(2.0, 1.0 - tf.cast(z, tf.float32))
        #gi3 *= norm
        gi3 = tf.transpose(gi3, perm=(1,0,2)) #nat,nzeta,nrad*nspec
        gi3 = tf.reshape(gi3, [nat, -1])

        #
        #g_i_l1l2 = tf.einsum('ijk,ijl->ijkl', g_ilxlylz, g_ilxlylz) #nat, nrad*nspec,n_lxlylz,n_lxlylz
        g_i_l1l2 = tf.expand_dims(g_ilxlylz,-1) * tf.expand_dims(g_ilxlylz,-2)
        g_i_l1l2 = tf.reshape(g_i_l1l2, [nat, -1, n_lxlylz*n_lxlylz]) #nat, nrad*nspec,n_lxlylz*n_lxlylz

        #rad_ij contains 2*zata + 1 radial functions

        lxlylz_sum2 = tf.reshape(self.lxlylz_sum[None,:] + self.lxlylz_sum[:,None], [-1])
        fact_norm2 = tf.reshape(self.fact_norm[None,:] * self.fact_norm[:, None], [-1])
        radial_ij_expanded = tf.gather(radial_ij, lxlylz_sum2, axis=2) # npair, nspec*nrad, n_lxlylz * n_lxlylz

        #g_ij_l1_plus_l2 = tf.einsum('ik,il->ikl',g_ij_lxlylz, g_ij_lxlylz) # npair,n_lxlylz,n_lxlylz
        g_ij_l1_plus_l2 = tf.expand_dims(g_ij_lxlylz,-1) * tf.expand_dims(g_ij_lxlylz,-2) # npair,n_lxlylz,n_lxlylz
        g_ij_l1_plus_l2 = tf.reshape(g_ij_l1_plus_l2, [-1, n_lxlylz*n_lxlylz])

        #g_ij_ll = tf.einsum('ijk,ik->ijk',radial_ij_expanded, g_ij_l1_plus_l2) #(npair, nrad*nspec,n_lxlylz*n_lxlylz)
        g_ij_ll = radial_ij_expanded * tf.expand_dims(g_ij_l1_plus_l2, 1)

        #g_ij_ll = tf.einsum('ijk,il->ijkl',g_ij_ll_rad, g_ij_lxlylz) # n_lxlylz,n_lxlylz,npair, nrad*nspec
        #contribution after summing over j
        g_i_l1_plus_l2 = tf.math.unsorted_segment_sum(data=g_ij_ll,
                                        segment_ids=first_atom_idx,num_segments=nat)#nat x nrad*nspec,n_lxlylz,n_lxlylz
        
        g_i_l1l2_ijk = tf.transpose(g_i_l1l2 * g_i_l1_plus_l2, [2,0,1]) * fact_norm2[:,None,None] #n_lxlylz * n_lxlylz, nat, nrad*nspec

        #g_i_l1l2_ijk = tf.einsum('ijk,ijk->kij',g_i_l1l2, g_i_l1_plus_l2) # n_lxlylz * n_lxlylz, nat, nrad*nspec

        nzeta2 = self.nzeta * self.nzeta

        g_i_l1l2 = tf.math.unsorted_segment_sum(data=g_i_l1l2_ijk,
                                        segment_ids=lxlylz_sum2, num_segments=nzeta2) # nzeta2, nat, nrad*nspec
        g_i_l1l2 = tf.transpose(g_i_l1l2, perm=[1,0,2])

        #g_i_ll_ijk2 = tf.transpose(g_i_ll_ijk2, perm=[1,0,2])
        gi4 = tf.reshape(g_i_l1l2, [nat, -1])

        return [gi3,gi4] # there are three combinations for th for body terms
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                ]
                 )
    def compute_charges(self, Vij, E1, E2, atomic_q0,total_charge):
        '''comput charges through the solution of linear system'''

        #this is removed after padding Vij with 1's at the last row and columns
        # Aij has exactly zero at N+1,N+1 elements a needed
        '''
        avoid using tf.pad padding altogether for memory efficiency
        E2 = tf.pad(E2, [[0,1]], constant_values=-1.0) 
        Aij = tf.pad(Vij, [[0,1],[0,1]], constant_values = 1.0)
        Aij += tf.linalg.diag(E2)
        '''

        N = tf.shape(E2)
        E2_padded = tf.concat([E2, [-1.0]], axis=0)  # shape [N+1]
        Aij = tf.concat([Vij, tf.ones(N,tf.float32)[:,None]], 1)
        Aij = tf.concat([Aij, tf.ones(N+1,tf.float32)[None,:]], 0)
        Aij += tf.linalg.diag(E2_padded)

        #Aij = 0.5 * (Aij + tf.transpose(Aij))

        '''
        #alternative implementation
        new_values = tf.constant([0.0], dtype=tf.float32)
        shape = tf.shape(Aij)
        index = tf.reshape(shape - 1, (1, 2))
        Aij = tf.tensor_scatter_nd_update(Aij, index, new_values)
        '''
        #E1 = tf.pad(-E1, [[0,1]], constant_values=total_charge)
        #total charge should be read from structures
        E1_padded = tf.concat([-E1, [total_charge]], axis=0)
        #atomic_q0 = tf.pad(atomic_q0, [[0,1]], constant_values=0.0)
        atomic_q0 = tf.concat([atomic_q0, [0.0]], axis=0)

        charges = tf.linalg.solve(Aij, E1_padded[:,None])
        '''
        lin_op_A = tf.linalg.LinearOperatorFullMatrix(
            Aij,
            is_self_adjoint=True,
            is_positive_definite=True,  # Optional: set to True if you know it is
            is_non_singular=True        # Optional: set to True if you know it is
        )
        outs = tf.linalg.experimental.conjugate_gradient(
            lin_op_A,
            E1,
            preconditioner=None,
            x=atomic_q0,
            tol=1e-05,
            max_iter=50,
            name='conjugate_gradient'
            )
        #outs[0]= max_iter, outs[2]=residual,outs[3]=basis vectors, outs[4]=preconditioner 
        charges = outs[1]
        '''
        return tf.reshape(charges, [-1])[:-1]
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                ]
                 )
    def compute_coulumb_energy(self,charges, atomic_q0, E1, E2, Vij):
        '''compute the coulumb energy
        Energy = \sum_i E_1i q_i + E_2i q^2/2 + \sum_i\sum_j Vij qiqj / 2
        '''
        q = charges
        dq = q - atomic_q0
        dq2 = dq * dq
        q_outer = q[:,None] * q[None,:]
        E = E1 * dq + 0.5 * (E2 * dq2 + tf.reduce_sum(Vij * q_outer, axis=-1))
        return tf.reduce_sum(E)

    @tf.function(jit_compile=False,
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
                tf.TensorSpec(shape=(), dtype=tf.float32)
                )])
    def tf_predict_energy_forces(self,x):
        ''' 
        x = (batch_species_encoder,positions,
                    nmax_diff, batch_nats,cells,C6,
                first_atom_idx,second_atom_idx,shift_vectors,num_pairs)
        '''
        rc = tf.constant(self.rcut,dtype=tf.float32)
        Nrad = tf.constant(self.Nrad, dtype=tf.int32)
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
        gaussian_width = x[10][:nat]
        chi0 = x[11][:nat]
        J0 = x[12][:nat]
        atomic_q0 = x[13][:nat]
        total_charge = x[14]
        #positions = tf.Variable(positions)
        #cell = tf.Variable(cell)
        if self.efield is not None:
            _efield = tf.constant(self.efield)
        else:
            _efield = tf.constant([0.,0.,0.])
        #with tf.GradientTape(persistent=True) as tape1:
            #the computation of Zstar is a second derivative and require additional gradient tape recording when computing the forces
       #     tape1.watch(_efield)
        with tf.GradientTape(persistent=True) as tape0:
            #'''
            tape0.watch(positions)
            tape0.watch(cell)

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

               # species_encoder_extended = tf.einsum('ik,il->ikl',
               #                                  tf.gather(species_encoder,first_atom_idx),
               #                                  tf.gather(species_encoder,second_atom_idx)
               #                                  )
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
            #radial_ij = tf.einsum('ijl,ik->ijkl',bf_radial, species_encoder_ij) # npairs x Nrad x nembeddingxzeta (l=zeta)
            #radial_ij = bf_radial[:,:,None,:] * species_encoder_ij[:,None,:,None]
            radial_ij = tf.expand_dims(bf_radial, 2) * tf.expand_dims(tf.expand_dims(species_encoder_ij, 1), -1)
            #radial_ij = tf.einsum('ijl,ik->ijkl',bf_radial, species_encoder_ij) # npairs x Nrad x nembeddingxzeta (l=zeta)
            radial_ij = tf.reshape(radial_ij, [num_pairs, self.Nrad*self.spec_size,self.number_radial_components])
            atomic_descriptors = tf.math.unsorted_segment_sum(data=radial_ij[:,:,0],
                                                              segment_ids=first_atom_idx, num_segments=nat) 

            #implement angular part: compute vect_rij dot vect_rik / rij / rik
            #rij_unit = tf.einsum('ij,i->ij',all_rij, 1.0 / (all_rij_norm+reg)) #npair,3
            rij_unit = all_rij / (tf.expand_dims(all_rij_norm + reg, -1))

            if self.body_order == 3:
                radial_ij_extended = tf.gather(radial_ij[:,:,:self.nzeta], self.lxlylz_sum, axis=2)
                Gi3 = self._to_three_body_terms(self.zeta, rij_unit, radial_ij_extended, first_atom_idx,nat)
                atomic_descriptors = tf.concat([atomic_descriptors, Gi3], axis=1)
            elif self.body_order == 4:
                  # this is require for proper backward propagation
                Gi3,Gi4 = self._to_four_body_terms(self.zeta, rij_unit, radial_ij, first_atom_idx, nat)
                atomic_descriptors = tf.concat([atomic_descriptors, Gi3], axis=1)
                atomic_descriptors = tf.concat([atomic_descriptors, Gi4], axis=1)

            #the descriptors can be scaled
            #append initial atomic charges
            if self.coulumb:
                atomic_descriptors = tf.concat([atomic_descriptors, atomic_q0[:,None]], axis=1)
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])
            #feature_size = Nrad * nembedding + Nrad, 2*zeta+1, nembedding
            #if self._normalize:
            #    # perform atom by atom normalization
            #    G_mean = tf.math.reduce_mean(atomic_descriptors, axis=1)
            #    G_std = tf.math.reduce_std(atomic_descriptors, axis=1)
            #    atomic_descriptors -= G_mean[:,None]
            #    atomic_descriptors /= (G_std[:,None] + 1e-8)

            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            if self.coulumb or self.include_vdw:
                total_energy = tf.reduce_sum(_atomic_energies[:,0])
            else:
                total_energy = tf.reduce_sum(_atomic_energies)

            if self.include_vdw:
                # C6 has a shape of nat
                C6 = tf.nn.relu(_atomic_energies[:,1])
                C6_ij = tf.gather(C6,second_atom_idx) * tf.gather(C6,first_atom_idx) # npair,
                #na from C6_extended in 1xNeigh and C6 in nat
                C6_ij = tf.sqrt(C6_ij + 1e-16)
                evdw = help_fn.vdw_contribution((all_rij_norm, C6_ij,
                                                 self.rmin_u,
                                                 self.rmax_u,
                                                 self.rmin_d,
                                                 self.rmax_d))[0]
                total_energy += evdw

                C6 = tf.pad(C6,[[0,nmax_diff]])
            else:
                C6 = tf.pad(C6,[[0,nmax_diff]])
            if self.coulumb:
                E1 = _atomic_energies[:,-2]
                E1 += chi0
                #include atomic electronegativity(chi0) and hardness (J0)
                _b = tf.identity(E1)
                E2 = _atomic_energies[:,-1]
                #E2 = tf.nn.relu(_atomic_energies[:,-1])
                E2 += J0
                _b -= E2 * atomic_q0
                _ewald = ewald(positions, cell, nat,
                        gaussian_width,self.accuracy, None, self.pbc, _efield)
                Vij = _ewald.recip_space_term() if self.pbc else _ewald.real_space_term()
                if self.efield is not None:
                    field_kernel = _ewald.sawtooth_PE()
                    _b += field_kernel
                charges = self.compute_charges(Vij, _b, E2, atomic_q0, total_charge)
                ecoul = self.compute_coulumb_energy(charges, atomic_q0, E1, E2, Vij)
                if self.efield is not None:
                    efield_energy = tf.reduce_sum(charges * field_kernel)
                    ecoul += efield_energy
                total_energy += ecoul
                pad_rows = tf.zeros([nmax_diff], dtype=tf.float32)
                charges = tf.concat([charges, pad_rows], axis=0)
            else:
                charges = tf.zeros([nmax_diff+nat], dtype=tf.float32)

        #differentiating a scalar w.r.t tensors
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
        #born effective charges for direct differentiation
        #if self.efield is not None and not self.is_training:
        #    Zstar = tf.squeeze(tape1.jacobian(forces,_efield, experimental_use_pfor=True)) #This is in unit of electron charges
        #    Zstar = tf.reshape(Zstar, [nat,9])
        #    Zstar = tf.pad(-Zstar, paddings=[[0,nmax_diff],[0,0]])
            #Zstar = tf.reshape(Zstar, [-1])
        #else:
        #    Zstar = tf.zeros((nat+nmax_diff,9))
        pad_rows = tf.zeros([nmax_diff, tf.shape(forces)[1]], dtype=forces.dtype)
        forces = tf.concat([-forces, pad_rows], axis=0)

        #forces = tf.pad(-forces, paddings=[[0,nmax_diff],[0,0]], constant_values=0.0)
        return [total_energy, forces,C6, charges,stress]

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
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
                )])
    def map_fn_parallel(self, elements):
        out_signature = [
            tf.float32,  # energies
            tf.float32,  # forces
#            tf.float32,  # atomic_features
            tf.float32,  # C6
            tf.float32,  # charges
            tf.float32,  # stress
    #        tf.float32  # zstar
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
        species_one_hot_encoder = tf.one_hot(spec_identity-1, depth=self.nelement)
        self.trainable_species_encoder = self.species_nets(species_one_hot_encoder) # nspecies x nembedding
        #species_encoder = inputs[1].to_tensor(shape=(-1, nmax)) #contains atomic number per atoms for all element in a batch
        species_encoder = tf.reshape(data['atomic_number'][:,:nmax], (-1,nmax)) #contains atomic number per atoms for all element in a batch

        # Compare each element in species_encoder to atomic_numbers in spec_identity: It has one true value per row
        matches = tf.equal(species_encoder[..., tf.newaxis], tf.cast(spec_identity,dtype=tf.float32))  # [batch_size, nmaxx, nspecies]

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
        batch_species_encoder = tf.reshape(batch_species_encoder, [batch_size, nmax * self.nspec_embedding])

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
            #initial charges
            batch_atomic_q0 = tf.gather(tf.cast(self.oxidation_states,tf.float32), species_indices)
            shape = tf.shape(batch_atomic_q0)
            batch_atomic_q0 = tf.where(valid_mask,
                                     batch_atomic_q0,
                                     tf.zeros(shape))
            batch_total_charge = data['total_charge']
        else:
            batch_atomic_chi0 = tf.zeros((batch_size,nmax), dtype=tf.float32)
            batch_atomic_J0 = tf.zeros((batch_size,nmax), dtype=tf.float32)
            batch_atomic_q0 = tf.zeros((batch_size,nmax), dtype=tf.float32)
            batch_total_charge = tf.zeros(batch_size, dtype=tf.float32)

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
        gaussian_width = tf.cast(data['gaussian_width'], tf.float32)

        #inputs = tf.constant([1.0], dtype=tf.float32)[:,None]
        #self._lambda_weights = tf.reshape(self._lambda_weights_nets(inputs), [-1]) # produce n_lambda weights
        #lambda_batch = tf.tile(self._lambda_weights[None,:], [batch_size,1])

        elements = (batch_species_encoder,positions, 
                    nmax_diff, batch_nats,cells,C6, 
                first_atom_idx,second_atom_idx,shift_vectors,num_pairs, gaussian_width,
                    batch_atomic_chi0,batch_atomic_J0,batch_atomic_q0,batch_total_charge)

        # energies, forces, atomic_features, C6, charges, stress
        energies, forces, C6, charges, stress = self.map_fn_parallel(elements)
        '''
        out_signature = [
            tf.float32,  # energies
            tf.float32,  # forces
#            tf.float32,  # atomic_features
            tf.float32,  # C6
            tf.float32,  # charges
            tf.float32,  # stress
            #tf.float32  # zstar
        ]


        energies, forces, C6, charges, stress = tf.map_fn(self.tf_predict_energy_forces, elements,
                                     fn_output_signature=out_signature,
                                     parallel_iterations=self.batch_size)
        '''
        #outs = [energies, forces, atomic_features]
        outs = {'energy':energies,
                'forces':forces,
                #'features':atomic_features,
                'C6': C6,
                'charges': charges,
                'stress': stress
                #'Zstar':Zstar
                }
        return outs

    def compile(self, optimizer, loss, loss_f=None):
        super().compile()
        self.optimizer = optimizer
        self.loss_e = loss
        if loss_f:
            self.loss_f = loss_f
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        #inputs_target = unpack_data(data)

        #inputs = inputs_target[:9]

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

            #emse_loss = tf.reduce_mean((ediff)**2)

            fmse_loss = tf.map_fn(self.loss_f, 
                                  (batch_nats,target_f,forces), 
                                  fn_output_signature=tf.float32)
            fmse_loss = tf.reduce_mean(fmse_loss) # average over batch

            delta_n = tf.constant(self._start_swa, dtype=tf.int64) - self._train_counter
            _step_fn = self.step_fn(delta_n)
            _ecost = self.ecost + (self.ecost_swa - self.ecost) * _step_fn #=self.ecost if self._start_swa > self._train_counter and self.ecost_swa otherwise
            _fcost = self.fcost + (self.fcost_swa - self.fcost) * _step_fn #=self.fcost if self._start_swa > self._train_counter and self.fcost_swa otherwise

            _loss = _ecost * emse_loss
            _loss += _fcost * fmse_loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        #assert trainable_vars == self.trainable_weights
        #trainable_vars = self.trainable_weights
        
        gradients = tape.gradient(_loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        #self._train_counter.assign_add(1)



        metrics = {'tot_st': self._train_counter}
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

        #with writer.set_as_default():
        #'''
        with self.train_writer.as_default(step=self._train_counter):
        #with self.train_writer.as_default():

            tf.summary.scalar('1. Losses/1. Total',_loss, self._train_counter)
            tf.summary.scalar('1. Losses/2. Energy',emse_loss,self._train_counter)
            tf.summary.scalar('1. Losses/3. Forces',fmse_loss,self._train_counter)
            tf.summary.scalar('2. Metrics/1. RMSE/atom',rmse,self._train_counter)
            tf.summary.scalar('2. Metrics/2. MAE/atom',mae,self._train_counter)
            tf.summary.scalar('2. Metrics/3. RMSE_F',rmse_f,self._train_counter)
            tf.summary.scalar('2. Metrics/4. MAE_F',mae_f,self._train_counter)
            #tf.summary.histogram(f'3. angular terms: lambda',self._lambda_weights,self._train_counter)
            for idx, spec in enumerate(self.species_identity):
                tf.summary.histogram(f'3. encoding /{spec}',self.trainable_species_encoder[idx],self._train_counter)
            if self.include_vdw:
                tf.summary.histogram(f'4. C6 parameters',C6, self._train_counter)
            if self.coulumb:
                tf.summary.histogram(f'5. charges',charges, self._train_counter)

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
        with self.train_writer.as_default(step=self._train_counter):
            tf.summary.scalar('2. Metrics/1. V_RMSE/atom',rmse,self._train_counter)
            tf.summary.scalar('2. Metrics/2. V_MAE/atom',mae,self._train_counter)
            tf.summary.scalar('2. Metrics/3. V_RMSE_F',rmse_f,self._train_counter)
            tf.summary.scalar('2. Metrics/3. V_MAE_F',mae_f,self._train_counter)



        return {key: metrics[key] for key in metrics.keys()}


    def predict_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        #inputs_target = unpack_data(data)
        batch_nats = tf.cast(tf.reshape(data['natoms'], [-1]), tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
       # target_f = tf.reshape(inputs_target[10][:,:nmax*3], [-1, 3*nmax])
        outs = self(data, training=False)
        #outs = self(data, training=False)
        e_pred = outs['energy']
        forces = outs['forces']
        stress = outs['stress']
        #Zstar = outs['Zstar']
        #features = outs['features']
        C6 = outs['C6']
        charges = outs['charges']

        #target = tf.cast(tf.reshape(inputs_target[10], [-1]), tf.float32)
        #target_f = tf.reshape(inputs_target[11][:,:nmax*3], [-1, 3*nmax])
        target = tf.cast(tf.reshape(data['energy'], [-1]), tf.float32)
        target_f = tf.reshape(data['forces'][:,:nmax*3], [-1, 3*nmax])

#        e_pred, forces, _ = self(inputs_target[:9], training=True)  # Forward pass
        forces = tf.reshape(forces, [-1, nmax*3])
        target_f = tf.cast(target_f, tf.float32)

        ediff = (e_pred - target)

        '''

        mae = tf.reduce_mean(tf.abs(ediff / batch_nats))
        rmse = tf.sqrt(tf.reduce_mean((ediff/batch_nats)**2))

        fmse = tf.map_fn(help_fn.force_mse, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        fmse = tf.reduce_mean(fmse)

        mae_f = tf.map_fn(help_fn.force_mae, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        mae_f = tf.reduce_mean(mae_f)

        rmse_f = tf.sqrt(fmse)
        metrics = {}

        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})
        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})
        '''
        forces_ref = tf.reshape(target_f, [-1, nmax, 3])
        forces_pred = tf.reshape(forces, [-1, nmax, 3])

        #if self.coulumb:
        charges = tf.reshape(charges, [-1,nmax])
        return [target, e_pred, forces_ref, forces_pred, batch_nats, charges,stress]
        #return [target, e_pred, metrics, forces_ref, forces_pred, batch_nats,stress]
