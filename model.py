from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import mendeleev
from mendeleev import element
import math 
import itertools, os
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
from networks import Networks
import helping_functions as help_fn

from functools import partial
import ase
from ase.neighborlist import neighbor_list
from ase import Atoms
from unpack_tfr_data import unpack_data
from ewald import ewald
import warnings





class mBP_model(tf.keras.Model):
   
    def __init__(self, configs):
        
        #allows to use all the base class of tf.keras Model
        super().__init__()
        
        #self.loss_tracker = self.metrics.Mean(name='loss')
        #network section
        self.layer_sizes = configs['layer_sizes']
        self._activations = configs['activations']
        self._radial_layer_sizes = configs['radial_layer_sizes']
        self.species_layer_sizes = configs['species_layer_sizes']

        #features parameters
        self.rcut = configs['rc_rad']
        self.Nrad = int(configs['Nrad'])
        self.zeta = configs['zeta'] if type(configs['zeta'])==list \
                else list(range(1,int(configs['zeta'])+1))
        self.nzeta = len(self.zeta)
        print(f'we are sampling zeta as: {self.zeta} with count {self.nzeta}')
        self.body_order = configs['body_order']
        base_size = self.Nrad * (2*self.nzeta)
        self.feature_size = self.Nrad + base_size
        if self.body_order == 4:
            self.feature_size += self.Nrad * (3*self.nzeta)
        self.species_correlation = configs['species_correlation']
        self.species_identity = configs['species_identity'] # atomic number
        self.nspecies = len(self.species_identity)

        self.nspec_embedding = self.species_layer_sizes[-1]
        if self.species_correlation == 'tensor':
            self.spec_size = self.nspec_embedding*self.nspec_embedding
        else:
            self.spec_size = self.nspec_embedding
        self.feature_size *= self.spec_size
                
        species_activations = ['silu' for x in self.species_layer_sizes[:-1]]
        species_activations.append('linear')

        #optimization parameters
        self.batch_size = configs['batch_size']
        self.fcost = float(configs['fcost'])
        self.ecost = float(configs['ecost'])
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
        
        #if not self.species_layer_sizes:
        #    self.species_layer_sizes = [self.nspec_embedding]
        self.features = False
        if not self.features:     
            self.atomic_nets = Networks(self.feature_size, 
                    self.layer_sizes, self._activations, 
                    l1=self.l1, l2=self.l2) 

       # create a species embedding network with 1 hidden layer Nembedding x Nspecies
        self.species_nets = Networks(self.nelement, 
                self.species_layer_sizes, 
                species_activations, prefix='species_encoder')
        #self.species_nets = Networks(self.nelement, [self.nspec_embedding], ['tanh'], prefix='species_encoder')

        #radial network
        if self.learn_radial:
            self._radial_layer_sizes.append(self.Nrad*(1+self.nzeta))
            radial_activations = ['silu' for s in self._radial_layer_sizes]
            self.radial_funct_net = Networks(self.Nrad, self._radial_layer_sizes, 
                                         radial_activations, 
                                         l1=self.l1, l2=self.l2,
                                 bias_initializer='zeros',
                                 prefix='radial-functions')

    @tf.function(
                input_signature=[(
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(3,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                )])

    def compute_rij(self, x):

        positions = x[0]
        cell = x[1]
        first_atom_idx = x[2]
        second_atom_idx = x[3]
        shift_vector = x[4]
        species_encoder = x[5]
        C6 = x[6]

        depth = tf.shape(positions)[0]
        _one_hot_j = tf.one_hot(second_atom_idx, depth=depth)
        pos_j = tf.matmul(_one_hot_j, positions)
        _one_hot_i = tf.one_hot(first_atom_idx, depth=depth)
        pos_i = tf.matmul(_one_hot_i, positions)
        all_rij = pos_j - pos_i + tf.tensordot(shift_vector,cell,axes=1)

        all_rij_norm = tf.linalg.norm(all_rij, axis=-1)
        all_rij = tf.RaggedTensor.from_value_rowids(all_rij,
                                                    first_atom_idx).to_tensor(default_value=1e-8) #nat,nneighx3
        all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm,
                                                         first_atom_idx).to_tensor(default_value=1e-8)

        species_encoder_extended = tf.matmul(_one_hot_j, species_encoder) * tf.matmul(_one_hot_j, species_encoder)
        #C6_extended = tf.matmul(_one_hot_j, C6[:,None]) #nneigh, nembedding

        species_encoder_extended = \
                tf.RaggedTensor.from_value_rowids(species_encoder_extended,
                                                  first_atom_idx).to_tensor() # nat, nmax_neigh, nembedding
        
        #C6_extended = tf.RaggedTensor.from_value_rowids(C6_extended, first_atom_idx).to_tensor() # nat, nmax_neigh
        species_encoder_ij = tf.einsum('ijk,ik->ijk',species_encoder_extended, species_encoder) #nat, nneigh, nembedding
        C6_extended = C6
        return [all_rij, all_rij_norm, species_encoder_ij, C6_extended]
    @tf.function
    def compute_cosine_terms(self, n):
        lxlylz = tf.map_fn(help_fn.find_three_non_negative_integers, n,
                                   fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32),
                                   parallel_iterations=self.nzeta)
        lxlylz = tf.reshape(lxlylz, [-1,3])
        lxlylz_sum = tf.reduce_sum(lxlylz, axis=-1) # the value of n for each lx, ly and lz

        #compute normalizations n! / lx!/ly!/lz!
        nfact = tf.map_fn(help_fn.factorial, lxlylz_sum,
                          fn_output_signature=tf.float32,
                          parallel_iterations=self.nzeta) #computed for all n_lxlylz

        #lx!ly!lz!
        fact_lxlylz = tf.reshape(tf.map_fn(help_fn.factorial, tf.reshape(lxlylz, [-1]),
                                           fn_output_signature=tf.float32,
                                           parallel_iterations=self.nzeta), [-1,3],
                                 )
        fact_lxlylz = tf.reduce_prod(fact_lxlylz, axis=-1)
        return lxlylz, lxlylz_sum, nfact, fact_lxlylz

    @tf.function
    def compute_charges(self, Vij, E1, E2):
        '''comput charges through the solution of linear system'''
        
        Aij = Vij + tf.set_diag(tf.zeros_like(Vij), E2)
        Aij = tf.pad(Aij, [[0,1],[0,1]], constant_values = 1.0)
        new_values = tf.constant([0.0], dtype=tf.float32)
        shape = tf.shape(Aij)
        index = tf.reshape(shape - 1, (1, 2))
        Aij = tf.tensor_scatter_nd_update(Aij, index, new_value)
        charges = tf.linalg.solve(Aij, E1[:,None])
        return tf.reshape(charges, [-1])

    @tf.function
    def compute_coulumb_energy(self,charges, E1, E2, Vij):
        '''compute the coulumb energy
        Energy = \sum_i E_1i q_i + E_2i q^2/2 + \sum_i\sum_j Vij qiqj / 2
        '''
        q = charges
        q2 = q * q
        q_outer = q[:,None] * q[None,:]
        E = E1 * q + 0.5 * (E2 * q2 + tf.reduce_sum(Vij * q_outer, axis=-1))
        return tf.reduce_sum(E)

    @tf.function(
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
                #tf.TensorSpec(shape=(None,), dtype=tf.float32),
                #tf.TensorSpec(shape=(None,), dtype=tf.float32),
                )])

    def tf_predict_energy_forces(self,x):
        ''' 
        x = (batch_species_encoder,positions,
                    nmax_diff, batch_nats,cells,C6,
                first_atom_idx,second_atom_idx,shift_vectors,num_neigh)
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
        num_neigh = x[9]
        first_atom_idx = x[6][:num_neigh]
        second_atom_idx = x[7][:num_neigh]
        shift_vector = tf.cast(tf.reshape(x[8][:num_neigh*3], 
                                          [num_neigh,3]), tf.float32)
        if self.coulumb:
            gaussian_width = x[10][:nat]

        with tf.GradientTape() as g:
            #'''
            g.watch(positions)
            #based on ase 
            #nneigh x 3
            all_rij = tf.gather(positions,second_atom_idx) - \
                     tf.gather(positions,first_atom_idx) + \
                    tf.tensordot(shift_vector,cell,axes=1)

            all_rij = tf.RaggedTensor.from_value_rowids(all_rij,
                                                        first_atom_idx
                                                        ).to_tensor(
                                                                default_value=1e-8
                                                                ) #nat,nneigh,3
            all_rij_norm = tf.linalg.norm(all_rij, axis=-1) #nat, nneigh
            #all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm_ragged,
            #                                                 first_atom_idx,
            #                                                 ).to_tensor(
            #                                                         default_value=1e-8
            #                                                         )

            
            # species_encoder_j * species_encoder_i
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


            species_encoder_extended = tf.reshape(species_encoder_extended, 
                                                  [-1, self.spec_size]
                                                   )
            species_encoder_ij = \
                    tf.RaggedTensor.from_value_rowids(species_encoder_extended,
                                                      first_atom_idx).to_tensor()
             
            
            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x Nneigh x embedding
            #fcuts is nat x Nneigh and reshaped to nat x Nneigh
            #_Nneigh = tf.shape(all_rij_norm)
            #neigh = _Nneigh[1]
            kn_rad = tf.ones(Nrad,dtype=tf.float32)
            bf_radial = help_fn.bessel_function(all_rij_norm,
                                            rc,kn_rad,
                                            Nrad) #
            if self.learn_radial:
                bf_radial = tf.reshape(bf_radial, [-1,Nrad])
                bf_radial = self.radial_funct_net(bf_radial)
                bf_radial = tf.reshape(bf_radial, [nat, -1, Nrad, self.nzeta+1])
                radial_ij = tf.einsum('ijkl,ijm->ijkml',bf_radial, species_encoder_ij) # nat x Nneigh x Nrad x nembeddingxzeta (l=zeta)
                radial_ij = tf.reshape(radial_ij, [nat,-1,Nrad*self.spec_size,self.nzeta+1])
                atomic_descriptors = tf.reduce_sum(radial_ij[:,:,:,0], axis=1) # sum over neigh
            else:
                radial_ij = tf.einsum('ijk,ijm->ijkm',bf_radial, species_encoder_ij) # nat x Nneigh x Nrad x nembeddingxzeta (l=zeta)
                atomic_descriptors = tf.reduce_sum(radial_ij, axis=1) # sum over neigh
            #atomic_descriptors = tf.reduce_sum(radial_ij, axis=(1,-1)) / tf.cast(self.zeta+1, tf.float32) # sum over neigh and zeta
            #atomic_descriptors = tf.reshape(atomic_descriptors, 
            #                                [nat, Nrad*self.spec_size]
            #                                )
            
            #implement angular part: compute vect_rij dot vect_rik / rij / rik
            reg = 1e-20
            rij_unit = tf.einsum('ijk,ij->ijk',all_rij, 1.0 / (all_rij_norm + reg))

            #for zeta = 0, anly the radial part contribute 2(\sum_j Rij)**2
            #Gi3 = [2.0 * tf.pow(
            #    tf.reduce_sum(radial_ij[:,:,:,:,0], axis=1),
            #    2)] 
            @tf.function
            def _angular_terms(x):
                '''
                compute vectorize three-body computation

                '''
                #expansion index
                z = tf.cast(x[0][0], tf.int32)
                r_idx = tf.cast(x[1][0], tf.int32)

                n = tf.range(z+1, dtype=tf.int32)
                lxlylz, lxlylz_sum, nfact, fact_lxlylz = self.compute_cosine_terms(n)
                
                #include (1 - cos(theta) * cos(theta)) # This enables probing of theta around pi/2
                '''
                lxlylz = tf.map_fn(help_fn.find_three_non_negative_integers, n,
                                   fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32),
                                   parallel_iterations=self.nzeta)
                lxlylz = tf.reshape(lxlylz, [-1,3])
                lxlylz_sum = tf.reduce_sum(lxlylz, axis=-1) # the value of n for each lx, ly and lz

                ###################
                #include (1 - cos(theta) * cos(theta)) # This enables probing of theta around pi/2
                n2 = tf.range(0,2*z+1,2, dtype=tf.int32)
                lxlylz2 = tf.map_fn(help_fn.find_three_non_negative_integers, n,
                                   fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32),
                                   parallel_iterations=self.nzeta)
                lxlylz2 = tf.reshape(lxlylz2, [-1,3])
                lxlylz2_sum = tf.reduce_sum(lxlylz2, axis=-1) # the value of n for each lx, ly and lz
                '''
                #rx^lx * ry^ly * rz^lz
                #this need to be regularized to avoid undefined derivatives
                rij_lxlylz = tf.pow(rij_unit[:,:,None,:] + 1e-16, tf.cast(lxlylz, tf.float32)[None,None,:,:])
                g_ij_lxlylz = tf.reduce_prod(rij_lxlylz, axis=-1) #nat x neigh x n_lxlylz

                nfact_lxlylz = nfact / fact_lxlylz # n_lxlylz

                #compute zeta! / (zeta-n)! / n!

                zeta_fact = help_fn.factorial(z)
                zeta_fact_n = tf.map_fn(help_fn.factorial, z-lxlylz_sum,
                                        fn_output_signature=tf.float32,
                                        parallel_iterations=self.nzeta
                                        )
                z_float = tf.cast(z, tf.float32)

                zetan_fact = zeta_fact / (zeta_fact_n * nfact)

                fact_norm = nfact_lxlylz * zetan_fact


                g_ij_lxlylz = tf.einsum('ijk,k->ijk',g_ij_lxlylz, fact_norm) # shape=(nat, neigh, n_lxlylz)

                if self.learn_radial:
                    g_ilxlylz = tf.einsum('ijk,ijl->ikl',radial_ij[:,:,:,r_idx], g_ij_lxlylz) #shape=(nat,nrad*species,n_lxlylz)
                else:
                    g_ilxlylz = tf.einsum('ijk,ijl->ikl',radial_ij, g_ij_lxlylz) #shape=(nat,nrad*species,n_lxlylz)
                g2_ilxlylz = g_ilxlylz * g_ilxlylz
                gi3p = tf.reduce_sum(g2_ilxlylz, axis=-1)

                _lambda_minus = tf.pow(-1.0, tf.cast(lxlylz_sum,tf.float32))
                gi3n = tf.einsum('ijk,k->ij',g2_ilxlylz,_lambda_minus) #nat,nrad*nspec

                norm = tf.pow(2.0 , 1. - z_float)
                #j==k term should be removed, lambda=-1 contribute nothing
                #R_j_equal_k = tf.reduce_sum(radial_ij[:,:,:,:,z] * radial_ij[:,:,:,:,z], axis=1) #nat, nrad, nspec
                #G_jk = 2 * R_j_equal_k 
                if self.body_order == 4:
                    # here we have four possible combination of lambda [(1,1), (1,-1), (-1,1), (-1,-1)] but only three of them are unique
                    #contribution after summing over k and l
                    g_ilxlylz_lambda_minus = tf.einsum('ijk,k->ijk',g_ilxlylz,_lambda_minus)
                     
                    g_i_ll_kl_pp = tf.einsum('ijk,ijl->ijkl',g_ilxlylz, g_ilxlylz)
                    g_i_ll_kl_np = tf.einsum('ijk,ijl->ijkl',g_ilxlylz_lambda_minus, g_ilxlylz)
                    g_i_ll_kl_nn = tf.einsum('ijk,ijl->ijkl',g_ilxlylz_lambda_minus, g_ilxlylz_lambda_minus)

                    
                    g_ij_ll = tf.einsum('ijm,ijn->ijmn',g_ij_lxlylz, g_ij_lxlylz)
                    #contribution after summing over j
                    if self.learn_radial:
                        g_i_ll_j = tf.einsum('ijk,ijmn->ikmn',radial_ij[:,:,:,r_idx], g_ij_ll) #nat nrad*nspec,n_lxlylz,n_lzlylz
                    else:
                        g_i_ll_j = tf.einsum('ijk,ijmn->ikmn',radial_ij, g_ij_ll) #nat nrad*nspec,n_lxlylz,n_lzlylz

                    #sum over the last two axes
                    # the normalization should be 2**z * 2**z * 2 so that the values are bound by 2 like the 3 body them
                    _norm = norm
                    _norm *= norm
                    _norm *= 0.5
                    gi4pp = tf.reduce_sum(g_i_ll_kl_pp * g_i_ll_j, axis=(2,3)) * _norm #nat,nrad*nspec
                    gi4np = tf.reduce_sum(g_i_ll_kl_np * g_i_ll_j, axis=(2,3)) * _norm
                    gi4nn = tf.reduce_sum(g_i_ll_kl_nn * g_i_ll_j, axis=(2,3)) * _norm
                    return [gi3p * norm, gi3n * norm, gi4pp, gi4np, gi4nn]



                return [gi3p * norm, gi3n * norm]
            #batch the radial term to have zeta components first

            #radial_terms = tf.cast([radial_ij[:,:,:,i] 
            #                        for i in range(1,self.nzeta+1)], tf.float32)
            
            if self.body_order == 3:
                #Because the computation cost for element of the loop is very different,
                #it might be faster to use a standard lood rather than vectorization.

                Gp = []
                Gn = []
                for i, z in enumerate(self.zeta):
                    gp,gn = _angular_terms(([z], [i+1]))
                    Gp.append(gp)
                    Gn.append(gn)

                #Gp,Gn = tf.map_fn(_angular_terms, (tf.constant(self.zeta)[:,None], tf.range(1,self.nzeta+1)[:,None]),
                #              fn_output_signature=[tf.float32,tf.float32],
                #              parallel_iterations=self.nzeta)
                Gi3 = tf.concat([Gp,Gn], 0) 
                #this is equivalent to 
                #for lambda in [-1,1]; for z in range(1, zeta+1)

                Gi3 = tf.transpose(Gi3, perm=(1,2,0)) 
                #'''
                body_descriptor_3 = tf.reshape(Gi3, [nat,-1])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_3], axis=1)

            elif self.body_order == 4:
                #G3p,G3n,G4pp,G4np,G4nn = tf.map_fn(_angular_terms, (tf.constant(self.zeta)[:,None], tf.range(1,self.nzeta+1)[:,None]),
                #              fn_output_signature=[tf.float32,tf.float32,tf.float32,tf.float32,tf.float32],
                #              parallel_iterations=self.nzeta)
                #Because the computation cost for element of the loop is very different,
                #it might be faster to use a standard lood rather than vectorization.
                G3p = []
                G3n = []
                G4pp = []
                G4np = []
                G4nn = []
                for i, z in enumerate(self.zeta):
                    gp,gn,g4pp,g4np,g4nn = _angular_terms(([z], [i+1]))
                    G3p.append(gp)
                    G3n.append(gn)
                    G4pp.append(g4pp)
                    G4np.append(g4np)
                    G4nn.append(g4nn)

                Gi4 = tf.concat([G3p,G3n,G4pp,G4np,G4nn], 0) 
                #this is equivalent to 
                #for lambda in [-1,1]; for z in range(1, zeta+1)

                Gi4 = tf.transpose(Gi4, perm=(1,2,0)) 
                #'''
                body_descriptor_4 = tf.reshape(Gi4, [nat,-1])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_4], axis=1)

            #feature_size = Nrad * nembedding + Nrad, 2*zeta+1, nembedding
            #the descriptors can be scaled
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])
            
            atomic_features = tf.reshape(atomic_descriptors, [-1])
            if self.features:
                return atomic_features
            #atomic_descriptors -= self.mean_descriptors
            #atomic_descriptors /= (self.std_descriptors + 1e-8)

            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            if self.include_vdw and not self.coulumb:
                # C6 has a shape of nat
                C6 = tf.nn.relu(_atomic_energies[:,1])
                atomic_energies = _atomic_energies[:,0]
                C6_ij = tf.gather(C6,second_atom_idx) * tf.gather(C6,first_atom_idx)
                C6_ij = \
                    tf.RaggedTensor.from_value_rowids(C6_ij,
                                                      first_atom_idx).to_tensor()

                #nat x Nneigh from C6_extended in 1xNeigh and C6 in nat
                C6_ij = tf.sqrt(C6_ij + 1e-16)
                evdw = help_fn.vdw_contribution((all_rij_norm, C6_ij,
                                                 self.rmin_u,
                                                 self.rmax_u,
                                                 self.rmin_d,
                                                 self.rmax_d))[0]
            elif self.include_vdw and self.coulumb:
                E1 = _atomic_energies[:,1]
                E2 = tf.nn.relu(_atomic_energies[:,2])
                C6 = tf.nn.relu(_atomic_energies[:,3])

                _ewald = ewald(positions, cell, nat, 
                        gaussian_width,self.accuracy, None, self.pbc)
                if self.pbc:
                    Vij = _ewald.recip_space_term()
                else:
                    Vij = _ewald.real_space_term()
                charges = self.compute_charges(Vij, E1, E2)
                ecoul = self.compute_coulumb_energy(charges, E1, E2, Vij)
                atomic_energies = _atomic_energies[:,0]
                C6_ij = tf.gather(C6,second_atom_idx) * tf.gather(C6,first_atom_idx)
                C6_ij = \
                    tf.RaggedTensor.from_value_rowids(C6_ij,
                                                      first_atom_idx).to_tensor()

                #nat x Nneigh from C6_extended in 1xNeigh and C6 in nat
                C6_ij = tf.sqrt(C6_ij + 1e-16)
                evdw = help_fn.vdw_contribution((all_rij_norm, C6_ij,
                                                 self.rmin_u,
                                                 self.rmax_u,
                                                 self.rmin_d,
                                                 self.rmax_d))[0]

            elif self.coulumb and not self.include_vdw:
                E1 = _atomic_energies[:,1]
                E2 = tf.nn.relu(_atomic_energies[:,2])
                _ewald = ewald(positions, cell, nat,
                        gaussian_width,self.accuracy, None, self.pbc)
                if self.pbc:
                    Vij = _ewald.recip_space_term()
                else:
                    Vij = _ewald.real_space_term()
                charges = self.compute_charges(Vij, E1, E2)
                ecoul = self.compute_coulumb_energy(charges, E1, E2, Vij)
                atomic_energies = _atomic_energies[:,0]

            else:
                atomic_energies = _atomic_energies
    #            tf.debugging.check_numerics(evdw, message='Total_energy_vdw contains NaN')
         

            total_energy = tf.reduce_sum(atomic_energies)
            

            if self.include_vdw:
                total_energy += evdw
            if self.coulumb:
                total_energy += ecoul

            #forces = g.jacobian(total_energy, positions)
        forces = g.gradient(total_energy, positions)
        forces = tf.pad(-forces, paddings=[[0,nmax_diff],[0,0]], constant_values=0.0)
        #padding = tf.zeros((nmax_diff,3))
        #forces = tf.concat([forces, padding], 0)
        #forces = tf.zeros((nmax_diff+nat,3))   
        
        #return [tf.cast(total_energy, tf.float32), tf.cast(forces, tf.float32), tf.cast(atomic_features, tf.float32)]
        if self.include_vdw and not self.coulumb:
            C6 = tf.pad(C6,[[0,nmax_diff]])
            return [total_energy, forces, atomic_features,C6]
        elif self.include_vdw and self.coulumb
            C6 = tf.pad(C6,[[0,nmax_diff]])
            charges = tf.pad(charges,[[0,nmax_diff]])
            return [total_energy, forces, atomic_features,C6, charges]
        elif self.coulumb and not self.include_vdw
            charges = tf.pad(charges,[[0,nmax_diff]])
            return [total_energy, forces, atomic_features,charges]
        return [total_energy, forces, atomic_features]

    @tf.function
    def func_map(self,elements):
        return tf.map_fn(self.tf_predict_energy_forces, elements,
                                     fn_output_signature=[tf.float32, tf.float32, tf.float32],
                                     parallel_iterations=self.batch_size)

    def call(self, inputs, training=False):
        '''input has a shape of batch_size x nmax_atoms x feature_size'''
        # may be just call the energy prediction here which will be needed in the train and test steps
        # the input are going to be filename from which descriptors and targets are going to be extracted
        #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh]

        batch_size = tf.shape(inputs[0])[0] 
        # the batch size may be different from the set batchsize saved in varaible self.batch_size
        # because the number of data point may not be exactly divisible by the self.batch_size.

        # todo: we need to pass nmax if we use padded tensors
       #[0-positions,1-species_encoder,2-C6,3-cells,4-natoms,5-i,6-j,7-S,8-neigh, 9-energy,10-forces]
        batch_nats = tf.cast(tf.reshape(inputs[4], [-1]), tf.int32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        batch_nmax = tf.tile([nmax], [batch_size])
        nmax_diff = tf.reshape(batch_nmax - batch_nats, [-1])

        #positions and species_encoder are ragged tensors are converted to tensors before using them
        positions = tf.reshape(inputs[0][:,:nmax*3], (-1, nmax*3))

        spec_identity = tf.constant(self.species_identity, dtype=tf.int32) - 1 # atomic number-1
        species_one_hot_encoder = tf.one_hot(spec_identity, depth=self.nelement)
        self.trainable_species_encoder = self.species_nets(species_one_hot_encoder) # nspecies x nembedding
        '''
        #species_encoder = inputs[1].to_tensor(shape=(-1, nmax)) #contains atomic number per atoms for all element in a batch
        species_encoder = tf.reshape(inputs[1][:,:nmax], (-1,nmax)) #contains atomic number per atoms for all element in a batch
        
        #batch_species_encoder = tf.reshape(inputs[1][:,:nmax], (-1,nmax)) #contains atomic number per atoms for all element in a batch
        #species_one_hot_encoder = tf.one_hot(species_encoder-1, depth=self.nelement)
        #batch_species_encoder = self.species_nets(species_one_hot_encoder)

        #species_encoder = inputs[1] #contains atomic number per atoms for all element in a batch
        batch_species_encoder = tf.zeros([batch_size, nmax, self.nspec_embedding], dtype=tf.float32)
        # This may be implemented better but not sure how yet
        for idx, spec in enumerate(self.species_identity):
            values = tf.ones([batch_size, nmax, self.nspec_embedding],
                             dtype=tf.float32) * self.trainable_species_encoder[idx]
            batch_species_encoder += tf.where(
                    tf.equal(tf.tile(species_encoder[:,:,tf.newaxis], 
                                     [1,1,self.nspec_embedding]
                                     ),
                                    tf.cast(spec,tf.float32)
                             ),
                    values, tf.zeros([batch_size, nmax, self.nspec_embedding],
                                     dtype=tf.float32)
                    )
        batch_species_encoder = tf.reshape(batch_species_encoder, [-1,self.nspec_embedding*nmax])

        '''

        # Build lookup table
        # Create a -1 mapping first (everything invalid maps to -1)
        mapping_array = -tf.ones([self.nelement], dtype=tf.int32)
        # Set valid mappings
        for idx, atomic_num in enumerate(self.species_identity):
            mapping_array = tf.tensor_scatter_nd_update(
                mapping_array, [[atomic_num - 1]], [idx]
            )

        # Now mapping_array[atomic_number-1] gives the species index or -1 if invalid

        species_encoder = inputs[1][:, :nmax]  # (batch_size, nmax)
        species_idx = tf.gather(mapping_array, 
                                tf.maximum(tf.cast(species_encoder,tf.int32) - 1, 0))  # (batch_size, nmax)
        # mask invalid species
        valid_mask = tf.not_equal(species_encoder, 0)

        # Now we can safely gather
        safe_species_idx = tf.maximum(species_idx, 0)  # replace -1 with 0 temporarily
        batch_species_encoder = tf.gather(self.trainable_species_encoder, safe_species_idx)  # (batch_size, nmax, nspec_embedding)

        # Mask out invalid (padding) entries
        batch_species_encoder *= tf.expand_dims(tf.cast(valid_mask, tf.float32), axis=-1)

        # Final reshape
        batch_species_encoder = tf.reshape(batch_species_encoder, [batch_size, nmax * self.nspec_embedding])

       #[0-positions,1-species_encoder,2-C6,3-cells,4-natoms,5-i,6-j,7-S,8-neigh, 9-energy,10-forces]

        C6 = inputs[2]
        cells = tf.reshape(inputs[3], [-1, 9])
        #cells = inputs[3]

        #first_atom_idx = tf.cast(inputs[6].to_tensor(shape=(self.batch_size, -1)), tf.int32)
        num_neigh = tf.cast(tf.reshape(inputs[8], [-1]), tf.int32)
        neigh_max = tf.reduce_max(num_neigh)
        first_atom_idx = tf.cast(inputs[5], tf.int32)
        second_atom_idx = tf.cast(inputs[6], tf.int32)
        #shift_vectors = tf.reshape(tf.cast(inputs[7][:,:neigh_max*3],tf.int32), (-1, neigh_max*3))
        shift_vectors = tf.cast(inputs[7][:,:neigh_max*3],tf.int32)
        if self.coulumb:
            gaussian_width = tf.cast(x[10], tf.float32)
            elements = (batch_species_encoder,positions, 
                    nmax_diff, batch_nats,cells,C6, 
                first_atom_idx,second_atom_idx,shift_vectors,num_neigh, gaussian_width)

        else:
            elements = (batch_species_encoder,positions, 
                    nmax_diff, batch_nats,cells,C6, 
                first_atom_idx,second_atom_idx,shift_vectors,num_neigh)

        if self.features:
            features = tf.map_fn(self.tf_predict_energy_forces, elements, 
                                     fn_output_signature=tf.float32,
                                     parallel_iterations=self.batch_size)
            #energies, forces, atomic_features = self.func_map(elements)
            return features, self.feature_size
        if self.include_vdw and not self.coulumb:
            energies, forces, atomic_features, C6 = tf.map_fn(self.tf_predict_energy_forces, elements, 
                                     fn_output_signature=[tf.float32, tf.float32, tf.float32,tf.float32],
                                     parallel_iterations=self.batch_size)
            #energies, forces, atomic_features = self.func_map(elements)
            return energies, forces, atomic_features, C6
        if self.include_vdw and self.coulumb:
            energies, forces, atomic_features, C6, charges = tf.map_fn(self.tf_predict_energy_forces, elements, 
                                     fn_output_signature=[tf.float32, tf.float32, tf.float32,tf.float32,tf.float32],
                                     parallel_iterations=self.batch_size)
            #energies, forces, atomic_features = self.func_map(elements)
            return energies, forces, atomic_features, C6, charges
        if (not self.include_vdw) and self.coulumb:
            energies, forces, atomic_features, charges = tf.map_fn(self.tf_predict_energy_forces, elements, 
                                     fn_output_signature=[tf.float32, tf.float32, tf.float32,tf.float32],
                                     parallel_iterations=self.batch_size)
            #energies, forces, atomic_features = self.func_map(elements)
            return energies, forces, atomic_features, charges

        energies, forces, atomic_features = tf.map_fn(self.tf_predict_energy_forces, elements, 
                                     fn_output_signature=[tf.float32, tf.float32, tf.float32],
                                     parallel_iterations=self.batch_size)
        #energies, forces, atomic_features = self.func_map(elements)
        return energies, forces, atomic_features

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        inputs_target = unpack_data(data)
        #inputs = inputs_target[:9]

        #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh, energy,forces]
        #target = tf.cast(tf.reshape(inputs_target[9], [-1]), tf.float32)
        batch_nats = tf.cast(tf.reshape(inputs_target[4], [-1]), tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        
        #target_f = tf.reshape(inputs_target[7], [-1, 3*nmax])
        #target_f = tf.cast(target_f, tf.float32)

        with tf.GradientTape() as tape:
            if self.include_vdw and (not self.coulumb) :
                e_pred, forces, _,C6 = self(inputs_target[:9], training=True)  # Forward pass
                target = tf.reshape(inputs_target[9], [-1])
                target_f = inputs_target[10][:,:nmax*3]
            elif (self.include_vdw) and (self.coulumb) :
                e_pred, forces, _,C6, charges = self(inputs_target[:10], training=True)  # Forward pass
                target = tf.reshape(inputs_target[10], [-1])
                target_f = inputs_target[11][:,:nmax*3]
            elif (not self.include_vdw) and (self.coulumb) :
                e_pred, forces, _,charges = self(inputs_target[:10], training=True)  # Forward pass
                target = tf.reshape(inputs_target[10], [-1])
                target_f = inputs_target[11][:,:nmax*3]
            else:
                e_pred, forces, _ = self(inputs_target[:9], training=True)  # Forward pass
                target = tf.reshape(inputs_target[9], [-1])
                target_f = inputs_target[10][:,:nmax*3]

            ediff = (e_pred - target)
            forces = tf.reshape(forces, [-1, 3*nmax])
            emse_loss = tf.reduce_mean((ediff)**2)

            fmse_loss = tf.map_fn(help_fn.force_loss, 
                                  (batch_nats,target_f,forces), 
                                  fn_output_signature=tf.float32)
            fmse_loss = tf.reduce_mean(fmse_loss)

            loss = self.ecost * emse_loss
            loss += self.fcost * fmse_loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))


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
        metrics.update({'loss': loss})
        metrics.update({'energy loss': self.ecost*emse_loss})
        metrics.update({'force loss': self.fcost*fmse_loss})

        #with writer.set_as_default():
        with self.train_writer.as_default(step=self._train_counter):

            tf.summary.scalar('1. Losses/1. Total',loss,self._train_counter)
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

        return {key: metrics[key] for key in metrics.keys()}


    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs_target = unpack_data(data)
        #inputs = inputs_target[:9]

        #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh, energy,forces]
        if self.include_vdw and (not self.coulumb) :
            e_pred, forces, _,C6 = self(inputs_target[:9], training=True)  # Forward pass
            target = tf.reshape(inputs_target[9], [-1])
            target_f = inputs_target[10][:,:nmax*3]
        elif (self.include_vdw) and (self.coulumb) :
            e_pred, forces, _,C6, charges = self(inputs_target[:10], training=True)  # Forward pass
            target = tf.reshape(inputs_target[10], [-1])
            target_f = inputs_target[11][:,:nmax*3]
        elif (not self.include_vdw) and (self.coulumb) :
            e_pred, forces, _,charges = self(inputs_target[:10], training=True)  # Forward pass
            target = tf.reshape(inputs_target[10], [-1])
            target_f = inputs_target[11][:,:nmax*3]
        else:
            e_pred, forces, _ = self(inputs_target[:9], training=True)  # Forward pass
            target = tf.reshape(inputs_target[9], [-1])
            target_f = inputs_target[10][:,:nmax*3]
        #target = tf.cast(tf.reshape(inputs_target[9], [-1]), tf.float32)
        batch_nats = tf.cast(tf.reshape(inputs_target[4], [-1]), tf.float32)

        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)

        #target_f = tf.reshape(inputs_target[10][:,:nmax*3], [-1, 3*nmax])

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
        loss = rmse * rmse + self.fcost * fmse_loss

        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})
        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})
        with self.train_writer.as_default(step=self._train_counter):
            tf.summary.scalar('2. Metrics/1. V_RMSE/atom',rmse,self._train_counter)
            tf.summary.scalar('2. Metrics/2. V_MAE/atom',mae,self._train_counter)
            tf.summary.scalar('2. Metrics/3. V_RMSE_F',rmse_f,self._train_counter)
            tf.summary.scalar('2. Metrics/3. V_MAE_F',mae_f,self._train_counter)


        metrics.update({'loss': loss})

        return {key: metrics[key] for key in metrics.keys()}


    def predict_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs_target = unpack_data(data)
        #inputs = inputs_target[:9]

        #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh, energy,forces]

       # target = tf.cast(tf.reshape(inputs_target[9], [-1]), tf.float32)
        batch_nats = tf.cast(tf.reshape(inputs_target[4], [-1]), tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
       # target_f = tf.reshape(inputs_target[10][:,:nmax*3], [-1, 3*nmax])
        if self.include_vdw and (not self.coulumb) :
            e_pred, forces, _,C6 = self(inputs_target[:9], training=True)  # Forward pass
            target = tf.reshape(inputs_target[9], [-1])
            target_f = inputs_target[10][:,:nmax*3]
        elif (self.include_vdw) and (self.coulumb) :
            e_pred, forces, _,C6, charges = self(inputs_target[:10], training=True)  # Forward pass
            target = tf.reshape(inputs_target[10], [-1])
            target_f = inputs_target[11][:,:nmax*3]
        elif (not self.include_vdw) and (self.coulumb) :
            e_pred, forces, _,charges = self(inputs_target[:10], training=True)  # Forward pass
            target = tf.reshape(inputs_target[10], [-1])
            target_f = inputs_target[11][:,:nmax*3]
        else:
            e_pred, forces, _ = self(inputs_target[:9], training=True)  # Forward pass
            target = tf.reshape(inputs_target[9], [-1])
            target_f = inputs_target[10][:,:nmax*3]
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
        return [target, e_pred, metrics, tf.reshape(target_f, [-1, nmax, 3]),tf.reshape(forces, [-1, nmax, 3]), batch_nats]
