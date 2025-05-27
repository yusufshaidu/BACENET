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
class mBP_model(tf.keras.Model):
   
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
        self.zeta = configs['zeta'] if type(configs['zeta'])==list \
                else list(range(1,int(configs['zeta'])+1))
        self.nzeta = len(self.zeta)
        #print(f'we are sampling zeta as: {self.zeta} with count {self.nzeta}')
        self.body_order = configs['body_order']
        base_size = self.Nrad * (2 * self.nzeta)
        self.feature_size = self.Nrad + base_size
        if self.body_order == 4:
            #self.feature_size += self.Nrad * (2*self.nzeta * (self.nzeta+1)) #4 * nzeta * (nzeta+1)/2
            self.feature_size += self.Nrad * 3 * self.nzeta
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
        self.total_charge = configs['total_charge']
        
        #if not self.species_layer_sizes:
        #    self.species_layer_sizes = [self.nspec_embedding]
        self.features = configs['features']
        if self.coulumb:
            self.feature_size += 1
        self.atomic_nets = Networks(self.feature_size, 
                    layer_sizes, _activations, 
                    l1=self.l1, l2=self.l2, normalize=configs['normalize']) 
        #self.atomic_nets = atomic_nets
        #self.species_nets = species_nets
        #self.radial_funct_net = radial_funct_net

       # create a species embedding network with 1 hidden layer Nembedding x Nspecies
        #species_activations = ['silu' for x in self.species_layer_sizes[:-1]]
        species_activations = ['silu' for x in self.species_layer_sizes[:-1]]
        species_activations.append('linear')

        self.species_nets = Networks(self.nelement, 
                self.species_layer_sizes, 
                species_activations, prefix='species_encoder')

        #radial network
        # each body order learn single component of the radial functions
        #one radial function per components and different one for 3 and 4 body 
        if self.body_order == 3:
            self.number_radial_components = (self.nzeta + 1)
        if self.body_order == 4:
            self.number_radial_components = (2 * self.nzeta + 1)

        _radial_layer_sizes.append(self.number_radial_components * self.Nrad)
        radial_activations = ['silu' for s in _radial_layer_sizes[:-1]]
        radial_activations.append('silu')
        self.radial_funct_net = Networks(self.Nrad, _radial_layer_sizes, 
                                         radial_activations, 
                                         #l1=self.l1, l2=self.l2
                                 #bias_initializer='zeros',
                                 prefix='radial-functions')
        # Add a non‐trainable counter:
        '''
        self._train_counter = self.add_weight(
            name="train_counter",
            shape=(),
            dtype=tf.int64,
            trainable=False,
            initializer="zeros"
        )
        '''
    '''
    @tf.function
    def estimate_species_chi0_J0(self):
        IE = tf.constant([element(sym).ionenergies[1]
            for sym in self.species], dtype=tf.float32)
        EA = tf.constant([0.0 if not element(sym).electron_affinity else
            element(sym).electron_affinity for sym in self.species], dtype=tf.float32)

        species_hardness = IE - EA
        species_electronegativity = 0.5 * (IE + EA)
        return (species_electronegativity, species_hardness)
    '''
    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ])
    def parallel_map_fn_factorial(self, n):
        return tf.map_fn(help_fn.factorial, n,
                          fn_output_signature=tf.float32,
                          parallel_iterations=self.nzeta)

    @tf.function(input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ])
    def compute_cosine_terms(self, n):
        lxlylz = tf.map_fn(help_fn.find_three_non_negative_integers, n,
                                   fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32),
                                   parallel_iterations=self.nzeta)
        lxlylz = tf.reshape(lxlylz, [-1,3])
        lxlylz_sum = tf.reduce_sum(lxlylz, axis=-1) # the value of n for each lx, ly and lz

        #compute normalizations n! / lx!/ly!/lz!
        nfact = self.parallel_map_fn_factorial(lxlylz_sum)
        #nfact = tf.map_fn(help_fn.factorial, lxlylz_sum,
        #                  fn_output_signature=tf.float32,
        #                  parallel_iterations=self.nzeta) #computed for all n_lxlylz

        #lx!ly!lz!
        fact_lxlylz = tf.reshape(self.parallel_map_fn_factorial(tf.reshape(lxlylz,[-1])), [-1,3])
        #fact_lxlylz = tf.reshape(tf.map_fn(help_fn.factorial, tf.reshape(lxlylz, [-1]),
        #                                   fn_output_signature=tf.float32,
        #                                   parallel_iterations=self.nzeta), [-1,3],
        #                         )
        fact_lxlylz = tf.reduce_prod(fact_lxlylz, axis=-1)
        return lxlylz, lxlylz_sum, nfact, fact_lxlylz

    '''
    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ])
    def compute_cosine_terms(self,n_values):
        # Expand all possible (lx, ly, lz) for n from 0 to max_n
        max_n = tf.reduce_max(n_values)
        #n_values is a range up to a max
        #max_n = n_values[-1]

        # Generate all (lx, ly, lz) such that lx + ly + lz <= max_n
        lx = tf.range(max_n + 1)
        ly = tf.range(max_n + 1)
        lz = tf.range(max_n + 1)
        lx, ly, lz = tf.meshgrid(lx, ly, lz, indexing='ij')

        lx = tf.reshape(lx, [-1])
        ly = tf.reshape(ly, [-1])
        lz = tf.reshape(lz, [-1])
        lxlylz = tf.stack([lx, ly, lz], axis=1)  # (N, 3)

        lxlylz_sum = tf.reduce_sum(lxlylz, axis=1)  # Total l = lx + ly + lz

        # Filter only those with lx + ly + lz == n for any n in n_values
        #n_values = tf.convert_to_tensor(n_values)
        n_values = n_values
        valid_mask = tf.reduce_any(tf.equal(tf.expand_dims(lxlylz_sum, 1), n_values), axis=1)
        lxlylz = tf.boolean_mask(lxlylz, valid_mask)
        lxlylz_sum = tf.boolean_mask(lxlylz_sum, valid_mask)

        # Compute factorials up to max_n and cache them
        factorials = tf.concat([[1.0], tf.cast(tf.math.cumprod(tf.range(1, max_n + 1)), tf.float32)], axis=0)

        # n! = (lx + ly + lz)!
        nfact = tf.gather(factorials, lxlylz_sum)

        # lx! * ly! * lz!
        fact_lxlylz = (
            tf.gather(factorials, lxlylz[:, 0]) *
            tf.gather(factorials, lxlylz[:, 1]) *
            tf.gather(factorials, lxlylz[:, 2])
        )

        return lxlylz, lxlylz_sum, nfact, fact_lxlylz
    '''
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32)]
                 )
    def _angular_terms(self,z, rij_unit):
        '''
        compute vectorize three-body computation

        '''
        n = tf.range(z+1, dtype=tf.int32)
        lxlylz, lxlylz_sum, nfact, fact_lxlylz = self.compute_cosine_terms(n)

        #rx^lx * ry^ly * rz^lz
        #this need to be regularized to avoid undefined derivatives
        rij_lxlylz = (rij_unit[:,None,:] + 1e-12)**tf.cast(lxlylz, tf.float32)[None,:,:]
        g_ij_lxlylz = tf.reduce_prod(rij_lxlylz, axis=-1) #npairs x n_lxlylz
        ###################
        nfact_lxlylz = nfact / fact_lxlylz # n_lxlylz

        #compute zeta! / (zeta-n)! / n!

        zeta_fact = help_fn.factorial(z)
        zeta_fact_n = self.parallel_map_fn_factorial(z-lxlylz_sum)
        #zeta_fact_n = tf.map_fn(help_fn.factorial, z-lxlylz_sum,
        #                        fn_output_signature=tf.float32,
        #                        parallel_iterations=self.nzeta
        #                        )
        z_float = tf.cast(z, tf.float32)
        zetan_fact = zeta_fact / (zeta_fact_n * nfact)
        fact_norm = nfact_lxlylz * zetan_fact
        g_ij_lxlylz = tf.einsum('ij,j->ij',g_ij_lxlylz, fact_norm) # shape=(nat, neigh, n_lxlylz)
        return g_ij_lxlylz, lxlylz_sum

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32)]
                 )
    def _to_three_body_terms(self,z, r_idx, rij_unit, 
                             radial_ij, first_atom_idx, nat):
    #def _to_three_body_terms(self, x):
        '''
        compute vectorize three-body computation

        '''
        #z = x[0][0]
        #r_idx = x[1][0]
        #rij_unit = x[2]
        #radial_ij = x[3]
        #first_atom_idx = x[4]
        #nat = x[5][0]
        g_ij_lxlylz, lxlylz_sum = self._angular_terms(z, rij_unit)
        g_ilxlylz = tf.einsum('ij,ik->ijk',radial_ij[:,:,r_idx], g_ij_lxlylz) #shape=(npair,nrad*species,n_lxlylz)
        #sum over neighbors
        g_ilxlylz = tf.math.segment_sum(data=g_ilxlylz,
                                            segment_ids=first_atom_idx)
        # nat x nrad*nspec, n_lxlylz
        g2_ilxlylz = g_ilxlylz * g_ilxlylz
        #sum over z
        gi3p = tf.reduce_sum(g2_ilxlylz, axis=-1)

        norm = tf.pow(2.0 , 1. - tf.cast(z, tf.float32))
        _lambda_minus = tf.pow(-1.0, tf.cast(lxlylz_sum,tf.float32))
        gi3n = tf.einsum('ijk,k->ij',g2_ilxlylz,_lambda_minus) #npairs,nrad*nspec

        return [gi3p * norm, gi3n * norm]

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32)]
                 )
    def _to_four_body_terms(self,z, r_idx, rij_unit, radial_ij, first_atom_idx, nat):
        '''
        compute  up to four-body computation

        '''
        g_ij_lxlylz, lxlylz_sum = self._angular_terms(z, r_idx, rij_unit)
        g_ilxlylz = tf.einsum('ij,ik->ijk',radial_ij[:,:,r_idx], g_ij_lxlylz) #shape=(npair,nrad*species,n_lxlylz)
        g_ilxlylz = tf.math.segment_sum(data=g_ilxlylz,
                                            segment_ids=first_atom_idx)# nat x nrad*nspec, n_lxlylz
        g2_ilxlylz = g_ilxlylz * g_ilxlylz
        #Example: sum of pairwise quantities per atom
        gi3p = tf.reduce_sum(g2_ilxlylz, axis=-1)
        norm = tf.pow(2.0 , 1. - tf.cast(z, tf.float32))
        _lambda_minus = tf.pow(-1.0, tf.cast(lxlylz_sum,tf.float32))
        gi3n = tf.einsum('ijk,k->ij',g2_ilxlylz,_lambda_minus) #npairs,nrad*nspec
        #j==k term should be removed, lambda=-1 contribute nothing
        #R_j_equal_k = tf.reduce_sum(radial_ij[:,:,:,:,z] * radial_ij[:,:,:,:,z], axis=1) #nat, nrad, nspec
        #G_jk = 2 * R_j_equal_k
        # here we have four possible combination of lambda [(1,1), (1,-1), (-1,1), (-1,-1)] but only three of them are unique
        #contribution after summing over k and l
        g4_ij_lxlylz = tf.einsum('ij,ik->ijk',radial_ij[:,:,self.nzeta+r_idx], g_ij_lxlylz) #shape=(npair,nrad*species,n_lxlylz)
        g4_i_lxlylz = tf.math.segment_sum(data=g4_ij_lxlylz,
                                        segment_ids=first_atom_idx)
                                        # nat x nrad*nspec*n_lxlylz

        g4_i_lxlylz = tf.reshape(g4_i_lxlylz, [nat,self.Nrad*self.spec_size, -1])

        _lambda_plus = tf.ones_like(_lambda_minus)
        lambda_pm = tf.einsum('i,j->ij',_lambda_minus, _lambda_plus)
        lambda_mm = tf.einsum('i,j->ij',_lambda_minus, _lambda_minus)
        g_i_ll_kl = tf.einsum('ijk, ijl->ijkl',g4_i_lxlylz,g4_i_lxlylz)

        g_ij_ll = tf.einsum('ij,ik->ijk',g_ij_lxlylz, g_ij_lxlylz)
        #contribution after summing over j
        g_i_ll_j = tf.einsum('ij,ikl->ijkl',radial_ij[:,:,self.nzeta+r_idx], g_ij_ll) #npair nrad*nspec,n_lxlylz,n_lzlylz
        g_i_ll_j = tf.math.segment_sum(data=g_i_ll_j,
                                        segment_ids=first_atom_idx)
        #sum over the last two axes
        # the normalization should be 2**z * 2**z * 2 so that the values are bound by 2 like the 3 body them
        _norm = norm * norm * 0.5
        g_i_ll_ijk = g_i_ll_j * g_i_ll_kl
        gi4pp = tf.reduce_sum(g_i_ll_ijk, axis=(2,3)) * _norm #nat,nrad*nspec
        gi4np = tf.einsum('ijkl, kl->ij',g_i_ll_ijk, lambda_pm) * _norm
        gi4nn = tf.einsum('ijkl, kl->ij',g_i_ll_ijk, lambda_mm) * _norm
        return [gi3p * norm, gi3n * norm, gi4pp, gi4np, gi4nn]

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def compute_charges(self, Vij, E1, E2, atomic_q0):
        '''comput charges through the solution of linear system'''
        
        #this is removed after padding Vij with 1's at the last row and columns
        # Aij has exactly zero at N+1,N+1 elements a needed
        E2 = tf.pad(E2, [[0,1]], constant_values=-1.0) 
        Aij = tf.pad(Vij, [[0,1],[0,1]], constant_values = 1.0)
        Aij += tf.linalg.diag(E2)
        #Aij = 0.5 * (Aij + tf.transpose(Aij))

        '''
        new_values = tf.constant([0.0], dtype=tf.float32)
        shape = tf.shape(Aij)
        index = tf.reshape(shape - 1, (1, 2))
        Aij = tf.tensor_scatter_nd_update(Aij, index, new_values)
        '''
        E1 = tf.pad(-E1, [[0,1]], constant_values=self.total_charge)
        #atomic_q0 = tf.pad(atomic_q0, [[0,1]], constant_values=0.0)
        charges = tf.linalg.solve(Aij, E1[:,None])
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
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
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
        first_atom_idx = x[6][:num_pairs]
        second_atom_idx = x[7][:num_pairs]
        shift_vector = tf.cast(tf.reshape(x[8][:num_pairs*3], 
                                          [num_pairs,3]), tf.float32)
        gaussian_width = x[10][:nat]
        chi0 = x[11][:nat]
        J0 = x[12][:nat]
        atomic_q0 = x[13][:nat]
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

            if self.species_correlation=='tensor':
                species_encoder_extended = tf.einsum('ik,il->ikl',
                                                 tf.gather(species_encoder,first_atom_idx),
                                                 tf.gather(species_encoder,second_atom_idx)
                                                 )
            else:
                species_encoder_extended = tf.einsum('ik,ik->ik',
                                                 tf.gather(species_encoder,first_atom_idx),
                                                 tf.gather(species_encoder,second_atom_idx)
                                                 )


            species_encoder_ij = tf.reshape(species_encoder_extended, 
                                                  [-1, self.spec_size]
                                                   ) #npairs,spec_size

            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x Nneigh x embedding
            #fcuts is nat x Nneigh and reshaped to nat x Nneigh
            #_Nneigh = tf.shape(all_rij_norm)
            #neigh = _Nneigh[1]
            kn_rad = tf.ones(Nrad,dtype=tf.float32)
            bf_radial0 = help_fn.bessel_function(all_rij_norm,
                                            rc,kn_rad,
                                            self.Nrad) #

            bf_radial1 = tf.reshape(bf_radial0, [-1,self.Nrad])
            bf_radial2 = self.radial_funct_net(bf_radial1)
            bf_radial = tf.reshape(bf_radial2, [num_pairs, self.Nrad, self.number_radial_components])
            radial_ij = tf.einsum('ijl,ik->ijkl',bf_radial, species_encoder_ij) # npairs x Nrad x nembeddingxzeta (l=zeta)
            radial_ij = tf.reshape(radial_ij, [num_pairs, self.Nrad*self.spec_size,self.number_radial_components])
            atomic_descriptors = tf.math.segment_sum(data=radial_ij[:,:,0],
                                                              segment_ids=first_atom_idx) 

            #implement angular part: compute vect_rij dot vect_rik / rij / rik
            rij_unit = tf.einsum('ij,i->ij',all_rij, 1.0 / (all_rij_norm + reg)) #npair,3
            #rij_unit = all_rij / (all_rij_norm[:,None] + reg)

            if self.body_order == 3:

                # this is require for proper backward propagation
                r_idx = 1
                zeta = tf.constant(self.zeta)
                z = zeta[0]
                gp,gn = self._to_three_body_terms(z, r_idx, rij_unit, radial_ij, first_atom_idx, nat)
                Gi3 = [gp]
                Gi3.append(gn)
                # we should tf range with Tensorarray
                for i in range(1,self.nzeta):
                    #r_idx += 1
                    z = zeta[i]
                    gp,gn = self._to_three_body_terms(z, i+1, rij_unit, radial_ij, first_atom_idx, nat)
                    Gi3.append(gp)
                    Gi3.append(gn)


                #elements = (zeta, r_idx, rij_unit_batch, radial_ij_batch, first_atom_idx_batch, nat_batch)
                '''
                elements = (zeta, r_idx)
                Gp,Gn = tf.map_fn(_to_three_body_terms,
                                  elements,
                              fn_output_signature=[tf.float32,tf.float32],
                              parallel_iterations=self.nzeta)
                '''
                #for lambda in [-1,1]; for z in range(1, zeta+1)
                Gi3 = tf.stack(Gi3, axis=1)
                body_descriptor_3 = tf.reshape(Gi3, [nat, -1])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_3], axis=1)
            elif self.body_order == 4:
                  # this is require for proper backward propagation
                r_idx = 1
                zeta = tf.constant(self.zeta)
                z = zeta[0]
                gp,gn,gpp, gnp, gnn = self._to_four_body_terms(z, r_idx, rij_unit, radial_ij, first_atom_idx, nat)
                G3p = [gp]
                G3n = [gn]
                G4pp = [gpp]
                G4np = [gnp]
                G4nn = [gnn]
                for i in range(1,self.nzeta):
                    r_idx = i+1
                    z = zeta[i]
                    gp,gn,gpp,gnp,gnn = self._to_four_body_terms(z, r_idx, rij_unit, radial_ij, first_atom_idx, nat)
                    G3p.append(gp)
                    G3n.append(gn)
                    G4pp.append(gpp)
                    G4np.append(gnp)
                    G4nn.append(gnn)

                Gi4 = tf.stack([G3p,G3n,G4pp,G4np,G4nn], axis=-1)
                #this is equivalent to
                #for lambda in [-1,1]; for z in range(1, zeta+1)
                #'''
                body_descriptor_4 = tf.reshape(Gi4, [nat,-1])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_4], axis=1)

            #the descriptors can be scaled
            #append initial atomic charges
            if self.coulumb:
                atomic_descriptors = tf.concat([atomic_descriptors, atomic_q0[:,None]], axis=1)
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])
            #feature_size = Nrad * nembedding + Nrad, 2*zeta+1, nembedding

            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            if self.include_vdw:
                # C6 has a shape of nat
                C6 = tf.nn.relu(_atomic_energies[:,1])
                atomic_energies = _atomic_energies[:,0]
                C6_ij = tf.gather(C6,second_atom_idx) * tf.gather(C6,first_atom_idx) # npair,
                #na from C6_extended in 1xNeigh and C6 in nat
                C6_ij = tf.sqrt(C6_ij + 1e-16)
                evdw = help_fn.vdw_contribution((all_rij_norm, C6_ij,
                                                 self.rmin_u,
                                                 self.rmax_u,
                                                 self.rmin_d,
                                                 self.rmax_d))[0]

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
                charges = self.compute_charges(Vij, _b, E2, atomic_q0)
                ecoul = self.compute_coulumb_energy(charges, atomic_q0, E1, E2, Vij)
                if self.efield is not None:
                    efield_energy = tf.reduce_sum(charges * field_kernel)
                    ecoul += efield_energy
                atomic_energies = _atomic_energies[:,0]
            if not self.coulumb and not self.include_vdw:
                atomic_energies = _atomic_energies
            total_energy = tf.reduce_sum(atomic_energies)
            if self.include_vdw:
                total_energy += evdw
            if self.coulumb:
                total_energy += ecoul
        #differentiating a scalar w.r.t tensors
        forces = tape0.gradient(total_energy, positions)
        #only need g to be persistent
        dE_dh = tape0.gradient(total_energy, cell)
        V = tf.abs(tf.linalg.det(cell))
        #V = tf.abs(tf.tensordot(cell[0], tf.linalg.cross(cell[1], cell[2]),axes=1))
        # Stress: stress_{ij} = (1/V) \sum_k dE/dh_{ik} * h_{jk}
        stress = tf.linalg.matmul(dE_dh, cell, transpose_b=True) / V
        #born effective charges for direct differentiation
        #if self.efield is not None and not self.is_training:
        #    Zstar = tf.squeeze(tape1.jacobian(forces,_efield, experimental_use_pfor=True)) #This is in unit of electron charges
        #    Zstar = tf.reshape(Zstar, [nat,9])
        #    Zstar = tf.pad(-Zstar, paddings=[[0,nmax_diff],[0,0]])
            #Zstar = tf.reshape(Zstar, [-1])
        #else:
        #    Zstar = tf.zeros((nat+nmax_diff,9))
        C6 = tf.pad(C6,[[0,nmax_diff]])
        if self.coulumb:
            charges = tf.pad(charges,[[0,nmax_diff]])
        else:
            charges = tf.zeros_like(C6)
        forces = tf.pad(-forces, paddings=[[0,nmax_diff],[0,0]], constant_values=0.0)
        return [total_energy, forces,C6, charges,stress]

    ''' 
    @tf.function
    def map_fn_parallel(self, elements):
        out_signature = [
            tf.float32,  # energies
            tf.float32,  # forces
#            tf.float32,  # atomic_features
            tf.float32,  # C6
            tf.float32,  # charges
            tf.float32,  # stress
            tf.float32  # zstar
        ]


        return tf.map_fn(self.tf_predict_energy_forces, elements,
                                     fn_output_signature=out_signature,
                                     parallel_iterations=self.batch_size)

    '''
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
        elements = (batch_species_encoder,positions, 
                    nmax_diff, batch_nats,cells,C6, 
                first_atom_idx,second_atom_idx,shift_vectors,num_pairs, gaussian_width,
                    batch_atomic_chi0,batch_atomic_J0,batch_atomic_q0)

        # energies, forces, atomic_features, C6, charges, stress
        #energies, forces, C6, charges, stress, Zstar = self.map_fn_parallel(elements)
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
            fmse_loss = tf.reduce_mean(fmse_loss)

            _loss = self.ecost * emse_loss
            _loss += self.fcost * fmse_loss
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
        metrics.update({'energy loss': self.ecost*emse_loss})
        metrics.update({'force loss': self.fcost*fmse_loss})

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
            #tf.summary.histogram(f'4. angular terms: lambda 1',lambda1,self._train_counter)
            #tf.summary.histogram(f'4. angular terms: lambda 2',lambda2,self._train_counter)
            for idx, spec in enumerate(self.species_identity):
                tf.summary.histogram(f'3. encoding /{spec}',self.trainable_species_encoder[idx],self._train_counter)
            if self.include_vdw:
                tf.summary.histogram(f'4. C6 parameters',C6, self._train_counter)
            if self.coulumb:
                tf.summary.histogram(f'5. charges',charges, self._train_counter)
            if self.efield is not None:
                tf.summary.histogram(f'6. Born charges',Zstar, self._train_counter)

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
        C6 = charges = None
        if self.include_vdw:
            C6 = outs['C6']
        if self.coulumb:
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
        C6 = charges = None
        if self.include_vdw:
            C6 = outs['C6']
        if self.coulumb:
            charges = outs['charges']

        #target = tf.cast(tf.reshape(inputs_target[10], [-1]), tf.float32)
        #target_f = tf.reshape(inputs_target[11][:,:nmax*3], [-1, 3*nmax])
        target = tf.cast(tf.reshape(data['energy'], [-1]), tf.float32)
        target_f = tf.reshape(data['forces'][:,:nmax*3], [-1, 3*nmax])

#        e_pred, forces, _ = self(inputs_target[:9], training=True)  # Forward pass
        forces = tf.reshape(forces, [-1, nmax*3])
        target_f = tf.cast(target_f, tf.float32)

        ediff = (e_pred - target)


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
        forces_ref = tf.reshape(target_f, [-1, nmax, 3])
        forces_pred = tf.reshape(forces, [-1, nmax, 3])
        if self.coulumb:
            charges = tf.reshape(charges, [-1,nmax])
            return [target, e_pred, metrics, forces_ref, forces_pred, batch_nats, charges,stress]
        return [target, e_pred, metrics, forces_ref, forces_pred, batch_nats,stress]
