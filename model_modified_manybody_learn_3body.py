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

import ase
from ase.neighborlist import neighbor_list
from ase import Atoms
from unpack_tfr_data import unpack_data
import warnings





class mBP_model(tf.keras.Model):
   
    def __init__(self, layer_sizes, 
                rcut, species_identity, 
                batch_size,
                activations,
                Nrad,
                thetaN,zeta,
                fcost=0.0,
                ecost=1.0,
                nelement=128,
                train_writer=None,
                l1=0.0,l2=0.0,
                nspec_embedding=4,
                include_vdw=False,
                rmin_u=3.0,
                rmax_u=5.0,
                rmin_d=11.0,
                rmax_d=12.0,
                body_order=3,
                features=False,
                layer_normalize=False,
                thetas_trainable=True,
                species_layer_sizes=[], #the last layer is eforced to be equal nspec_embedding
                species_correlation='tensor',
                learn_angular_terms=False

                 ):
        
        #allows to use all the base class of tf.keras Model
        super().__init__()
        
        #self.loss_tracker = self.metrics.Mean(name='loss')

        self.layer_sizes = layer_sizes
        self.rcut = rcut
        self.species_identity = species_identity # atomic number
        self.batch_size = batch_size
        self._activations = activations
        self.Nrad = Nrad
        self.thetaN = thetaN
        self.zeta = float(zeta)
        self.body_order = body_order
        self.species_correlation = species_correlation
        base_size = self.Nrad * self.thetaN
        self.feature_size = self.Nrad + base_size
        self.nspec_embedding = nspec_embedding
        if self.species_correlation == 'tensor':
            self.spec_size = self.nspec_embedding*self.nspec_embedding
        else:
            self.spec_size = self.nspec_embedding

        if self.body_order >= 4:
            for k in range(3,self.body_order):
                self.feature_size += base_size
        self.feature_size *= self.spec_size
        self.features = features

        self.fcost = float(fcost)
        self.ecost = float(ecost)
        self.nspecies = len(self.species_identity)
        self.train_writer = tf.summary.create_file_writer(train_writer+'/train')
        self.l1 = l1
        self.l2 = l2
        self.include_vdw = include_vdw
        self.rmin_u = rmin_u
        self.rmax_u = rmax_u
        self.rmin_d = rmin_d
        self.rmax_d = rmax_d
        self.layer_normalize = layer_normalize
        self.tf_pi = tf.constant(math.pi, dtype=tf.float32)
        self.thetas_trainable = thetas_trainable
        self.learn_angular_terms = learn_angular_terms
        # the number of elements in the periodic table
        self.nelement = nelement
        
        self.species_layer_sizes = species_layer_sizes
        if not self.species_layer_sizes:
            self.species_layer_sizes = [self.nspec_embedding]
        if self.species_layer_sizes[-1] != self.nspec_embedding:
            warnings.warn(f'the last layer of species embedding is set to {nspec_embedding}')
            self.species_layer_sizes[-1] = self.nspec_embedding

        species_activations = ['tanh' for x in self.species_layer_sizes[:-1]]
        species_activations.append('linear')

        if not self.features:     
            self.atomic_nets = Networks(self.feature_size, self.layer_sizes, self._activations, l1=self.l1, l2=self.l2, normalize=self.layer_normalize)

       # create a species embedding network with 1 hidden layer Nembedding x Nspecies
        self.species_nets = Networks(self.nelement, self.species_layer_sizes, species_activations, prefix='species_encoder')
        #self.species_nets = Networks(self.nelement, [self.nspec_embedding], ['tanh'], prefix='species_encoder')

        if self.learn_angular_terms:
            self.ang_funct_nets = Networks(3, [128,128,128,self.thetaN], ['sigmoid', 'sigmoid','sigmoid','sigmoid'], 
                                 bias_initializer='zeros',
                                 prefix='angular-functions')
        self.radial_funct_net = Networks(self.Nrad, [128,128,128,self.Nrad], ['silu', 'silu','silu','linear'], 
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
        thetasN = tf.constant(self.thetaN, dtype=tf.int32)

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
        
        tstep = self.tf_pi / tf.cast(thetasN, tf.float32)
        theta_s = tf.range(1,thetasN+1, dtype=tf.float32) * tstep
        cos_theta_s = tf.cos(theta_s)
        sin_theta_s = tf.sin(theta_s)


        with tf.GradientTape() as g:
            #'''
            g.watch(positions)
            #based on ase 
            #nneigh x 3
            all_rij = tf.gather(positions,second_atom_idx) - \
                     tf.gather(positions,first_atom_idx) + \
                    tf.tensordot(shift_vector,cell,axes=1)

            all_rij_norm_ragged = tf.linalg.norm(all_rij, axis=-1) #nat, nneigh
            #all_rij = tf.RaggedTensor.from_value_rowids(all_rij,
            #                                            first_atom_idx
            #                                            ).to_tensor(
            #                                                    default_value=1e-8
            #                                                    ) #nat,nneigh,3
            all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm_ragged,
                                                             first_atom_idx,
                                                             ).to_tensor(
                                                                     default_value=1e-8
                                                                     )

            
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
             
            #C6_ij = tf.gather(C6,second_atom_idx) * tf.gather(C6,first_atom_idx)
            #C6_ij = \
            #        tf.RaggedTensor.from_value_rowids(C6_ij,
            #                                          first_atom_idx).to_tensor()

            #all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm,
            #                                                 first_atom_idx).to_tensor(default_value=1e-8)

            #create an n_neigh x nats and fill all valid j index with 1.
            #when multiplied by the positions, correctly pick up the atoms positions of the atom

            '''
            all_rij, all_rij_norm, \
                    species_encoder_ij, C6_extended = \
                    self.compute_rij((positions,cell,
                    first_atom_idx,
                    second_atom_idx,
                    shift_vector,
                    species_encoder, C6))
            if self.include_vdw:
                #nat x Nneigh from C6_extended in 1xNeigh and C6 in nat
                C6_ij = tf.sqrt(C6_ij)
                evdw = self.vdw_contribution((all_rij_norm, C6_ij))[0]
    #            tf.debugging.check_numerics(evdw, message='Total_energy_vdw contains NaN')

            '''
            #this is no needed because have no contribution from atoms outside cutoff because of fc
            #mask the non-zero values after ragged tensor is converted to fixed shape
            # This is very important to avoid adding contributions from atoms outside the sphere
            #nonzero_val = tf.where(all_rij_norm > 1e-8, 1.0,0.0)
            #nonzero_val = tf.cast(nonzero_val, tf.float32)
            #gauss_args = gauss_args * nonzero_val[:,tf.newaxis,:]



            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x Nneigh x embedding
            #fcuts is nat x Nneigh and reshaped to nat x Nneigh
            #gauss_term = tf.reshape(help_fn.tf_app_gaussian(tf.reshape(gauss_args, [-1])), tf.shape(gauss_args))
            #fcut = tf.reshape(help_fn.tf_fcut(tf.reshape(all_rij_norm, [-1]), rc), tf.shape(all_rij_norm))
            #arg = self.tf_pi / rc * tf.einsum('l,ij->ijl',tf.range(1, Nrad+1, dtype=tf.float32) * kn_rad, all_rij_norm)
            #nkn_rad = tf.range(1, Nrad+1, dtype=tf.float32) * kn_rad
            #arg = self.tf_pi / rc * nkn_rad[None,None,:] * all_rij_norm[:,:,None]
            #r = tf.einsum('l,ij->ijl',tf.ones(Nrad), all_rij_norm)
            #r = all_rij_norm[:,:,None]
            _Nneigh = tf.shape(all_rij_norm)
            Nneigh = _Nneigh[1]
            kn_rad = tf.ones(Nrad,dtype=tf.float32)
            bf_radial = tf.reshape(help_fn.bessel_function(all_rij_norm,
                                                           rc,kn_rad,
                                                           Nrad), [nat*Nneigh, Nrad])
            bf_radial = self.radial_funct_net(bf_radial)
            bf_radial = tf.reshape(bf_radial, [nat, Nneigh, Nrad])
            radial_ij = tf.einsum('ijk,ijl->ijkl',bf_radial, species_encoder_ij) # nat x Nneigh x Nrad x nembedding**2
            atomic_descriptors = tf.reduce_sum(radial_ij, axis=1) # sum over neigh
            atomic_descriptors = tf.reshape(atomic_descriptors, 
                                            [nat, Nrad*self.spec_size]
                                            )

            #implement angular part: compute vect_rij dot vect_rik / rij / rik
            reg = 1e-20
            rij_unit = tf.einsum('ij,i->ij',all_rij, 1.0 / (all_rij_norm_ragged + reg))
            rij_unit = tf.RaggedTensor.from_value_rowids(rij_unit,
                                                        first_atom_idx
                                                        ).to_tensor(
                                                                default_value=1e-8
                                                                ) #nat,nneigh,3



            #nat x Nneigh X Nneigh
            #rij.rik
            #cos_theta_ijk = tf.matmul(rij_unit, tf.transpose(rij_unit, (0,2,1)))
            cos_theta_ijk = tf.einsum('ijk,ilk->ijl', rij_unit, rij_unit) #nat x Nneigh X Nneigh

            #do we need to remove the case of j=k?
            #i==j or i==k already contribute 0 because of 1/2**zeta or  constant,
            #this is probably not correct if zeta is too small. We assume it is true for now
            #lmn = tf.shape(rij_dot_rik)
            #l = lmn[0]
            #m = lmn[1]
            #n = lmn[2]
            #extract results outside i=j=k
            #row_col_dep = tf.transpose(tf.where(rij_dot_rik!=1e20))
            #row, col, dep = tf.transpose(tf.where(rij_dot_rik!=1e20))
            #row = tf.reshape(row_col_dep[0], (l,m,n))
            #col = tf.reshape(row_col_dep[1], (l,m,n))
            #dep = tf.reshape(row_col_dep[2], (l,m,n))
            #cond = tf.where(tf.logical_and(tf.logical_and(row!=col, col!=dep), row!=dep))
            #cos_theta_ijk = tf.reshape(tf.RaggedTensor.from_value_rowids(cos_theta_ijk, cond[:,0]).to_tensor(default_value=1e-8), [nat,-1])

            #cos_theta_ijk = tf.reshape(cos_theta_ijk, [nat,Nneigh*Nneigh])
            
            #clip values to avoid divergence in the derivative with the division by sin(theta)
            reg2 = 1e-6

            cos_theta_ijk = tf.clip_by_value(cos_theta_ijk, clip_value_min=-1.0+reg2, clip_value_max=1.0-reg2)

            
            #nat x thetasN x N_unique
            if not self.learn_angular_terms:

                cos_theta_ijk2 = cos_theta_ijk * cos_theta_ijk
                sin_theta_ijk = tf.sqrt(1.0 - cos_theta_ijk2)
                _cos_theta_ijk_theta_s = tf.einsum('ijk,l->ijkl',cos_theta_ijk,cos_theta_s) #nat x Nneigh X Nneigh x thetasN
                _sin_theta_ijk_theta_s = tf.einsum('ijk,l->ijkl',sin_theta_ijk,sin_theta_s) #nat x Nneigh X Nneigh x thetasN
                cos_theta_ijk_theta_s = 1.0 + _cos_theta_ijk_theta_s + _sin_theta_ijk_theta_s #nat x Nneigh X Nneigh x thetasN
                #tf.debugging.check_numerics(cos_theta_ijk_theta_s, message='cos_theta_ijk_theta_s contains NaN')

                norm_ang = 2.0

                #Nat,Nneigh, Nneigh,ThetasN
                #note that the factor of 2 in BP functions cancels the 1/2 due to double counting
                cos_theta_ijk_theta_s_zeta = tf.pow(cos_theta_ijk_theta_s / norm_ang, self.zeta) #nat x Nneigh X Nneigh x thetasN
            else:
                cos_theta_ijk = tf.reshape(cos_theta_ijk, [-1])
                cos_theta_ijk2 = cos_theta_ijk * cos_theta_ijk
                sin_theta_ijk = tf.sqrt(1.0 - cos_theta_ijk2)
                angular_inputs = tf.stack([tf.ones_like(cos_theta_ijk, 
                                                   dtype=tf.float32), 
                                           cos_theta_ijk, 
                                           sin_theta_ijk], axis=1) #[nat * Nneigh * Nneigh, 3]

                #angular_inputs = tf.reshape(angular_inputs, [-1,3])
                
                cos_theta_ijk_theta_s_zeta = self.ang_funct_nets(angular_inputs) # expected to be nat*Nneigh*Nneighxthetas
                cos_theta_ijk_theta_s_zeta = tf.reshape(cos_theta_ijk_theta_s_zeta, 
                                                        [nat,Nneigh,Nneigh,thetasN])

            # I am now using the same radial functions for angular and radial functions
            #bf_radial = tf.reshape(help_fn.bessel_function(all_rij_norm,
            #                                               rc,kn_ang,
            #                                               Nrad), [nat, Nneigh, Nrad])
            #radial_ij = tf.einsum('ijk,ijl->ijkl',bf_radial, species_encoder_ij) # nat x Nneigh x Nrad x nembedding**2
            ##############################################
            #radial_ij = tf.einsum('ijk,ijl->ijkl', bf_radial, species_encoder_ij)

            # dimension = [Nat, Nneigh, Nneigh, Nrad, thetaN]
            #exp_ang_theta_ijk = cos_theta_ijk_theta_s_zeta[:,:,:,tf.newaxis,:] * radial_ij[:,:,tf.newaxis,:,tf.newaxis]
            Base_vector_ij_s = tf.einsum('ijkl,ijmn->ikmln',cos_theta_ijk_theta_s_zeta, radial_ij) # Nat, Nneigh, Nrad,thetaN, nembedding**2
            ang_size = Nrad * thetasN * self.spec_size

            body_descriptor_3 = tf.einsum('ijklm,ijkm->iklm',Base_vector_ij_s, radial_ij) #shape=(nat,Nrad,thetaN,self.nspec_embedding)
            body_descriptor_3 = tf.reshape(body_descriptor_3, [nat,ang_size])
            atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_3], axis=1)

            if self.body_order >= 4:
                body_tensor_4 = Base_vector_ij_s * Base_vector_ij_s
                body_descriptor_4 = tf.einsum('ijklm,ijkm->iklm',body_tensor_4, radial_ij)
                body_descriptor_4 = tf.reshape(body_descriptor_4, [nat, ang_size])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_4], axis=1)
            if self.body_order >= 5:
                body_tensor_5 = body_tensor_4 * Base_vector_ij_s
                body_descriptor_5 = tf.einsum('ijklm,ijkm->iklm',body_tensor_5, radial_ij)
                body_descriptor_5 = tf.reshape(body_descriptor_5, [nat, ang_size])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_5], axis=1)
            


            #feature_size = Nrad,nembedding + Nrad, thetasN, nembedding
            #the descriptors can be scaled
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])
            
            atomic_features = tf.reshape(atomic_descriptors, [-1])
            if self.features:
                return atomic_features
            #atomic_descriptors -= self.mean_descriptors
            #atomic_descriptors /= (self.std_descriptors + 1e-8)

            #predict energy and forces
            atomic_energies = self.atomic_nets(atomic_descriptors)
            total_energy = tf.reduce_sum(atomic_energies)
            

            #if self.include_vdw:
            #    total_energy += evdw

            forces = g.jacobian(total_energy, positions)
            #forces = g.gradient(total_energy, positions)
        forces = tf.pad(-forces, paddings=[[0,nmax_diff],[0,0]], constant_values=0.0)
        #padding = tf.zeros((nmax_diff,3))
        #forces = tf.concat([forces, padding], 0)
        #forces = tf.zeros((nmax_diff+nat,3))   
        
        #return [tf.cast(total_energy, tf.float32), tf.cast(forces, tf.float32), tf.cast(atomic_features, tf.float32)]
        return [total_energy, forces, atomic_features]


    def call(self, inputs, training=False):
        '''input has a shape of batch_size x nmax_atoms x feature_size'''
        # may be just call the energy prediction here which will be needed in the train and test steps
        # the input are going to be filename from which descriptors and targets are going to be extracted
        #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh]

        batch_size = tf.shape(inputs[0])[0] 
        # the batch size may be different from the set batchsize saved in varaible self.batch_size
        # because the number of data point may not be exactly divisible by the self.batch_size.

        # todo: we need to pass nmax if we use padded tensors
        batch_nats = tf.cast(tf.reshape(inputs[4], [-1]), tf.int32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        batch_nmax = tf.tile([nmax], [batch_size])
        nmax_diff = tf.reshape(batch_nmax - batch_nats, [-1])

        #positions and species_encoder are ragged tensors are converted to tensors before using them
        positions = tf.reshape(inputs[0][:,:nmax*3], (-1, nmax*3))
        #positions = tf.reshape(inputs[0], (-1, 3*nmax))
        #positions = tf.cast(positions, dtype=tf.float32)
        #obtain species encoder

        spec_identity = tf.constant(self.species_identity, dtype=tf.int32) - 1 # atomic number-1
        species_one_hot_encoder = tf.one_hot(spec_identity, depth=self.nelement)
        self.trainable_species_encoder = self.species_nets(species_one_hot_encoder) # nspecies x nembedding
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
        C6 = inputs[2]
        cells = tf.reshape(inputs[3], [-1, 9])
        #cells = inputs[3]

        #first_atom_idx = tf.cast(inputs[6].to_tensor(shape=(self.batch_size, -1)), tf.int32)
        num_neigh = tf.cast(tf.reshape(inputs[8], [-1]), tf.int32)
        neigh_max = tf.reduce_max(num_neigh)
        first_atom_idx = tf.cast(inputs[5], tf.int32)
        second_atom_idx = tf.cast(inputs[6], tf.int32)
        shift_vectors = tf.reshape(tf.cast(inputs[7][:,:neigh_max*3],tf.int32), (-1, neigh_max*3))

        elements = (batch_species_encoder,positions, 
                    nmax_diff, batch_nats,cells,C6, 
                first_atom_idx,second_atom_idx,shift_vectors,num_neigh)

        if self.features:
            features = tf.map_fn(self.tf_predict_energy_forces, elements, 
                                     fn_output_signature=tf.float32,
                                     parallel_iterations=self.batch_size)
            return features, self.feature_size

        energies, forces, atomic_features = tf.map_fn(self.tf_predict_energy_forces, elements, 
                                     fn_output_signature=[tf.float32, tf.float32, tf.float32],
                                     parallel_iterations=self.batch_size)
        return energies, forces, atomic_features

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.

        inputs_target = unpack_data(data)
        #inputs = inputs_target[:9]

        #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh, energy,forces]
        #target = tf.cast(tf.reshape(inputs_target[9], [-1]), tf.float32)
        target = tf.reshape(inputs_target[9], [-1])
        batch_nats = tf.cast(tf.reshape(inputs_target[4], [-1]), tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        
        target_f = tf.reshape(inputs_target[10][:,:nmax*3], [-1, 3*nmax])
        #target_f = tf.reshape(inputs_target[7], [-1, 3*nmax])
        #target_f = tf.cast(target_f, tf.float32)

        with tf.GradientTape() as tape:
            e_pred, forces, _ = self(inputs_target[:9], training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            ediff = (e_pred - target)
            forces = tf.reshape(forces, [-1, 3*nmax])

            #emse_loss = tf.reduce_mean((ediff/batch_nats)**2)
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
            for idx, spec in enumerate(self.species_identity):
                tf.summary.histogram(f'3. encoding /{spec}',self.trainable_species_encoder[idx],self._train_counter)

        return {key: metrics[key] for key in metrics.keys()}


    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs_target = unpack_data(data)
        #inputs = inputs_target[:9]

        #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh, energy,forces]
        target = tf.cast(tf.reshape(inputs_target[9], [-1]), tf.float32)
        batch_nats = tf.cast(tf.reshape(inputs_target[4], [-1]), tf.float32)

        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)

        target_f = tf.reshape(inputs_target[10][:,:nmax*3], [-1, 3*nmax])

        e_pred, forces, _ = self(inputs_target[:9], training=True)  # Forward pass
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

        target = tf.cast(tf.reshape(inputs_target[9], [-1]), tf.float32)
        batch_nats = tf.cast(tf.reshape(inputs_target[4], [-1]), tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        target_f = tf.reshape(inputs_target[10][:,:nmax*3], [-1, 3*nmax])
        e_pred, forces, _ = self(inputs_target[:9], training=True)  # Forward pass
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
