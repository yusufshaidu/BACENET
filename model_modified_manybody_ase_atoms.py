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


class mBP_model(tf.keras.Model):
   
    def __init__(self, layer_sizes, rcut, species_identity, 
                 width, batch_size,
                activations,
                rc_ang,RsN_rad, RsN_ang,
                thetaN,width_ang,zeta,
                fcost=0.0,
                ecost=1.0,
                pbc=True,
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
                min_radial_center=0.5,
                species_out_act='linear',
                mean_descriptors=None,
                std_descriptors=None,
                features=False,
                layer_normalize=False):
        
        #allows to use all the base class of tf.keras Model
        super().__init__()
        
        #self.loss_tracker = self.metrics.Mean(name='loss')

        self.layer_sizes = layer_sizes
        self.rcut = rcut
        self.species_identity = species_identity # atomic number
        self._width = width
        self.batch_size = batch_size
        self._activations = activations
        self.rcut_ang = rc_ang
        self.RsN_rad = RsN_rad
        self.RsN_ang = RsN_ang
        self.thetaN = thetaN
        self.zeta = float(zeta)
        self.width_ang = float(width_ang)
        self.body_order = body_order
        # self.RsN_rad must be equal to self.RsN_ang
        base_size = self.RsN_rad * self.thetaN
        self.feature_size = self.RsN_rad + base_size
        if self.body_order >= 4:
            for k in range(3,self.body_order):
                self.feature_size += base_size
        self.nspec_embedding = nspec_embedding
        self.feature_size *= self.nspec_embedding
        self.mean_descriptors = mean_descriptors
        self.std_descriptors = std_descriptors
        self.features = features

        self.pbc = pbc
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
        self.min_radial_center = min_radial_center
        self.species_out_act = species_out_act
        self.layer_normalize = layer_normalize
        # the number of elements in the periodic table
        self.nelement = nelement
        if not self.features:     
            self.atomic_nets = Networks(self.feature_size, self.layer_sizes, self._activations, l1=self.l1, l2=self.l2, normalize=self.layer_normalize)

       # create a species embedding network with 1 hidden layer Nembedding x Nspecies
       #self.species_nets = Networks(self.nelement, [self.nspec_embedding], ['tanh',self.species_out_act], prefix='species_encoder')
       #self.species_nets = Networks(self.nelement, [self.nspec_embedding], ['tanh'], prefix='species_encoder')
       #self.species_nets = Networks(self.nelement, [self.nspec_embedding], [self.species_out_act], prefix='species_encoder')
        self.species_nets = Networks(self.nelement, [self.nelement,self.nspec_embedding], ['tanh',self.species_out_act], prefix='species_encoder')

        constraint = None
        Nwidth_rad = self.RsN_rad
        Nwidth_ang = self.RsN_ang
        init = tf.keras.initializers.RandomNormal(mean=3, stddev=0.05)
        self.rbf_nets = Networks(1, [self.RsN_rad], ['sigmoid'], 
                                 weight_initializer=init,
                                 bias_initializer='zeros',
                                 prefix='rbf')
        init = tf.keras.initializers.RandomNormal(mean=3, stddev=0.05)
        self.rbf_nets_ang = Networks(1, [self.RsN_ang], ['sigmoid'], 
                                     weight_initializer=init,
                                     bias_initializer='zeros',
                                     prefix='rbf_ang')
        
        ''' 
        init = tf.keras.initializers.RandomNormal(mean=self._width, stddev=0.05)
        self.width_nets = Networks(1, [Nwidth_rad], ['softplus'],
                                  weight_initializer=init,
                                  bias_initializer='zeros',
                                  kernel_constraint=constraint,
                                  bias_constraint=constraint, prefix='radial_width')
        init = tf.keras.initializers.RandomNormal(mean=self.width_ang, stddev=0.05)
        self.width_nets_ang = Networks(1, [Nwidth_ang], ['softplus'],
                                  weight_initializer=init,
                                  bias_initializer='zeros',
                                  kernel_constraint=constraint,
                                  bias_constraint=constraint,prefix='ang_width')

        constraint = None
        init = tf.keras.initializers.GlorotNormal(seed=12345)
        self.Rs_rad_nets = Networks(1, [self.RsN_rad], ['sigmoid'],
                                      weight_initializer=init,
                                      bias_initializer=init,
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='Rs_rad')
        init = tf.keras.initializers.GlorotNormal(seed=34567)
        self.Rs_ang_nets = Networks(1, [self.RsN_ang], ['sigmoid'],
                                      weight_initializer=init,
                                      bias_initializer=init,
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='Rs_ang')
        
        init = tf.keras.initializers.RandomNormal(mean=self.zeta, stddev=0.05)
        self.zeta_nets = Networks(1, [1], ['softplus'],
                                  weight_initializer=init,
                                  bias_initializer='zeros',
                                  kernel_constraint=constraint,
                                  bias_constraint=constraint, prefix='zeta')
        '''
        init = tf.keras.initializers.GlorotNormal(seed=56789)
        self.thetas_nets = Networks(1, [self.thetaN], ['sigmoid'],
                                      weight_initializer=init,
                                      bias_initializer=init,
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='thetas')

    #@tf.function(
                #input_signature=[(
                #tf.TensorSpec(shape=(None,), dtype=tf.float32),
                #tf.TensorSpec(shape=(None,), dtype=tf.float32),
                #tf.TensorSpec(shape=(), dtype=tf.int32),
                #tf.TensorSpec(shape=(), dtype=tf.int32),
                #tf.TensorSpec(shape=(None,), dtype=tf.float32),
                #tf.TensorSpec(shape=(None,), dtype=tf.float32),
                #tf.TensorSpec(shape=(3,3), dtype=tf.float32),
                #tf.TensorSpec(shape=(None,), dtype=tf.int32),
                #tf.TensorSpec(shape=(None,), dtype=tf.float32),
                #tf.TensorSpec(shape=(None,), dtype=tf.float32),
                #tf.TensorSpec(shape=(None,), dtype=tf.float32),
                #tf.TensorSpec(shape=(None,), dtype=tf.float32),
                #)]
                #)

    @tf.function()
    def tf_predict_energy_forces(self,x):
        ''' 
        elements = (batch_species_encoder, batch_kn_rad,
                nmax_diff, batch_nats,
                batch_zeta, batch_kn_ang,
                batch_theta_s, atoms)

        '''
        rc = tf.constant(self.rcut,dtype=tf.float32)
        Ngauss = tf.constant(self.RsN_rad, dtype=tf.int32)
        #rc_ang = tf.constant(self.rcut_ang, dtype=tf.float32)
        rc_ang = tf.constant(self.rcut, dtype=tf.float32)
        #Ngauss_ang = tf.constant(self.RsN_ang, dtype=tf.int32)
        Ngauss_ang = tf.constant(self.RsN_rad, dtype=tf.int32)
        thetasN = tf.constant(self.thetaN, dtype=tf.int32)



        nat = tf.cast(x[3], dtype=tf.int32)
        nmax_diff = tf.cast(x[2], dtype=tf.int32)
        species_encoder = tf.reshape(x[0][:nat*self.nspec_embedding], [nat,self.nspec_embedding])

        atoms = x[7]

        #width = tf.cast(x[1], dtype=tf.float32)
        kn_rad = x[1]
        #positions = tf.reshape(atoms.positions, [nat,3])

        #zeta cannot be a fraction
        zeta = x[4]
        #width_ang = x[6]
        kn_ang = x[5]
        
        cell = tf.cast(atoms.cell,tf.float32)

        #replica_idx = x[8]
        #Rs = x[9]
        #Rs_ang = x[10]
        theta_s = x[6]

        evdw = 0.0

#        if self.include_vdw:
        C6 = tf.cast(atoms.get_array('C6'), tf.float32)

        sin_theta_s = tf.sin(theta_s)
        cos_theta_s = tf.cos(theta_s)
#        sin_theta_s2 = sin_theta_s * sin_theta_s
        if self.mean_descriptors is None or self.std_descriptors is None:
            self.mean_descriptors = tf.zeros((nat, self.feature_size), dtype=tf.float32)
            self.std_descriptors = tf.ones((nat, self.feature_size), dtype=tf.float32)
        
        with tf.GradientTape() as g:

            positions = tf.cast(atoms.positions, tf.float32)
            g.watch(positions)
            #based on ase 
            first_atom_idx, second_atom_idx, shift_vector = neighbor_list('ijS', atoms, rcut)
            #nneigh x 3
            all_rij = tf.gather(positions,second_atom_idx) - tf.gather(positions,first_atom_idx) + shift_vector.dot(atoms.cell)
            all_rij_norm = tf.linalg.norm(rij, axis=-1) + 1e-8
            all_rij = tf.RaggedTensor.from_value_rowids(all_rij, first_atom_idx).to_tensor(default_value=1e-8)
            all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm, first_atom_idx).to_tensor(default_value=1e-8)

            #rj-ri
            #all_rij = positions_extended - positions[:, tf.newaxis, :] #shape=(nat, nneigh, 3)
            species_encoder_extended = tf.gather(species_encoder, second_atom_idx) #nneigh, nembedding
            C6_extended = tf.gather(C6, second_atom_idx) #nneigh, nembedding
            
            species_encoder_extended = tf.RaggedTensor.from_value_rowids(species_encoder_extended, 
                                                                         first_atom_idx).to_tensor() # nat, nmax_neigh, nembedding
            C6_extended = tf.RaggedTensor.from_value_rowids(C6_extended, first_atom_idx).to_tensor() # nat, nmax_neigh

            species_encoder_ij = tf.einsum('ijk->ik->ijk',species_encoder_extended, species_encoder) #nat, nneigh, nembedding
            
            '''
            if self.include_vdw:
                #nat x Nneigh from C6_extended in 1xNeigh and C6 in nat
                inball_vdw = tf.where(tf.logical_and(all_rij_norm <=self.rmax_d, all_rij_norm>1e-8))
                all_rij_norm = tf.gather_nd(all_rij_norm, inball_vdw)
                all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm, inball_vdw[:,0]).to_tensor(default_value=1e-8)
                C6ij = tf.sqrt(C6_extended * C6[:, tf.newaxis])
                C6ij = tf.gather_nd(C6ij, inball_vdw)
                C6ij = tf.RaggedTensor.from_value_rowids(C6ij, inball_vdw[:,0]).to_tensor()
                #the dimensions needs to be checked!!!!!!!!!
                #C6ij = tf.sqrt((C6[:, tf.newaxis] * C6[tf.newaxis, :]))
                evdw = self.vdw_contribution((all_rij_norm, C6ij))[0]
    #            tf.debugging.check_numerics(evdw, message='Total_energy_vdw contains NaN')
            '''

            tf_pi = tf.constant(math.pi, dtype=tf.float32)
            arg = tf_pi / rc * tf.einsum('l,ij->ijl',tf.range(1, Ngauss+1, dtype=tf.float32) * kn_rad, all_rij_norm)
            #arg = tf.reshape(arg, [-1])
            r = tf.einsum('l,ij->ijl',tf.ones(Ngauss), all_rij_norm)
            #r = tf.reshape(r, [-1])
            bf_radial = tf.reshape(help_fn.bessel_function(r,arg,rc), [nat, -1, Ngauss])
            #bf_radial = tf.reshape(help_fn.bessel_function(r,arg,rc), [-1, Ngauss])
            
            atomic_descriptors = tf.einsum('ijk,ijl->ikl',bf_radial, species_encoder_ij)
            # sum over neighbors j including periodic boundary conditions
            #atomic_descriptors = tf.reduce_sum(tf.reshape(args,[nat,-1, self.nspec_embedding*Ngauss]), axis=1)
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, -1])

            #implement angular part: compute vect_rij dot vect_rik / rij / rik
                   
            #inball_ang = tf.where(all_rij_norm <=rc_ang)
            #all_rij_norm = tf.gather_nd(all_rij_norm, inball_ang)
            #produces a list of tensors with different shapes because atoms have different number of neighbors
            #all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm, inball_ang[:,0]).to_tensor(default_value=1e-8)
            
            #all_rij = tf.gather_nd(all_rij, inball_ang)
            #produces as list of tensors with different shapes because atoms have different number of neighbors
            #all_rij = tf.RaggedTensor.from_value_rowids(all_rij, inball_ang[:,0]).to_tensor(default_value=1e-8)
            
            #species_encoder_ij = tf.gather_nd(species_encoder_ij, inball_ang)
            #species_encoder_ij = tf.RaggedTensor.from_value_rowids(species_encoder_ij, inball_ang[:,0]).to_tensor()
            #species_encoder_ij = tf.reshape(species_encoder_ij, [-1,self.nspec_embedding])

            reg = 1e-20
            all_rij_norm_inv = 1.0 / (all_rij_norm + reg)
            
            #number of neighbors:
            #_Nneigh = tf.shape(all_rij)
            _Nneigh = tf.shape(all_rij_norm)
            Nneigh = _Nneigh[1]
            
            rij_unit = tf.einsum('ijk,ij->ijk',all_rij, all_rij_norm_inv)
            #nat x Nneigh X Nneigh
            #rij.rik
            #cos_theta_ijk = tf.matmul(rij_unit, tf.transpose(rij_unit, (0,2,1)))
            cos_theta_ijk = tf.einsum('ijk, ilk -> ijl', rij_unit, rij_unit) 

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

            cos_theta_ijk = tf.reshape(cos_theta_ijk, [nat,-1])
            
            #clip values to avoid divergence in the derivative with the division by sin(theta)
            reg2 = 1e-6

            cos_theta_ijk = tf.clip_by_value(cos_theta_ijk, clip_value_min=-1.0+reg2, clip_value_max=1.0-reg2)
            cos_theta_ijk2 = cos_theta_ijk * cos_theta_ijk
            #nat x thetasN x N_unique
            sin_theta_ijk = tf.sqrt(1.0 - cos_theta_ijk2)
            _cos_theta_ijk_theta_s = tf.einsum('ij,l->ijl',cos_theta_ijk, cos_theta_s) 
            _sin_theta_ijk_theta_s = tf.einsum('ij,l->ijl',sin_theta_ijk, sin_theta_s)
            cos_theta_ijk_theta_s = 1.0 + _cos_theta_ijk_theta_s + _sin_theta_ijk_theta_s
            #tf.debugging.check_numerics(cos_theta_ijk_theta_s, message='cos_theta_ijk_theta_s contains NaN')
            
            norm_ang = 2.0 
            #Nat,Nneigh, Nneigh,ThetasN
            #note that the factor of 2 in BP functions cancels the 1/2 due to double counting
            cos_theta_ijk_theta_s_zeta = tf.reshape((cos_theta_ijk_theta_s / norm_ang)**zeta, (nat,Nneigh,-1))

            #tf.debugging.check_numerics(cos_theta_ijk_theta_s_zeta, message='cos_theta_ijk_theta_s_zeta contains NaN')

            #rik_norm_rs = width_ang[tf.newaxis,tf.newaxis,:] * (all_rij_norm[:,:,tf.newaxis]  - Rs_ang[tf.newaxis,tf.newaxis,:])**2
            #rik_norm_rs = tf.einsum('ijk,k->ijk',(all_rij_norm[:,:,tf.newaxis]  - Rs_ang[tf.newaxis,tf.newaxis,:])**2, width_ang)

            #compute the exponetial term: Nat, Nneigh, RNs_ang
            #gauss_term = tf.reshape(help_fn.tf_app_gaussian(tf.reshape(rik_norm_rs, [-1])), tf.shape(rik_norm_rs))
            #fcut = tf.reshape(help_fn.tf_fcut(tf.reshape(all_rij_norm, [-1]), rc_ang), tf.shape(all_rij_norm))
            #'''
            #arg = tf_pi / rc_ang * tf.einsum('l,ij->ijl',tf.range(1, Ngauss_ang+1, dtype=tf.float32) * kn_ang, all_rij_norm)
            #arg = tf.reshape(arg, [-1])
            #r = tf.einsum('l,ij->ijl',tf.ones(Ngauss_ang), all_rij_norm)
            #r = tf.reshape(r, [-1])
            #bf_radial_ang = tf.reshape(help_fn.bessel_function(r,arg,rc_ang), [nat,-1, Ngauss_ang])
            #'''
            #radial_ij = tf.einsum('ijk,ij->ijk',gauss_term, fcut)
              
            #radial_ij = tf.einsum('ijk,ijl->ijkl',radial_ij,species_encoder_ij)
            #########################################
            # I am now using the same radial functions for angular and radial functions
            ##############################################
            radial_ij = tf.einsum('ijk,ijl->ijkl', bf_radial, species_encoder_ij)
            radial_ij = tf.reshape(radial_ij, [nat,Nneigh,Ngauss*self.nspec_embedding])

            # dimension = [Nat, Nneigh, Nneigh, Ngauss, thetaN]
            #exp_ang_theta_ijk = cos_theta_ijk_theta_s_zeta[:,:,:,tf.newaxis,:] * radial_ij[:,:,tf.newaxis,:,tf.newaxis]
            exp_ang_theta_ijk = tf.einsum('ijk,ijl->ikl',cos_theta_ijk_theta_s_zeta, radial_ij)
            Base_vector_ij_s = tf.reshape(exp_ang_theta_ijk, [nat,Nneigh,thetasN,-1])
            
            #Base_vector_ij_s = tf.reduce_sum(exp_ang_theta_ijk, axis=2)
            #dimension = [Nat, Nneigh, Ngauss_ang, thetaN]
            #Base_vector_ij_s = tf.reshape(exp_ang_theta_ijk, [nat*Nneigh, Ngauss_ang*self.nspec_embedding, -1])
            
            #radial_ij = tf.reshape(tf.tile(radial_ij[:,:,:,tf.newaxis],[1,1,1,thetasN]), [nat,Nneigh,-1])
            #body_descriptor_3 = tf.reduce_sum(Base_vector_ij_s * radial_ij, axis=1) #sum over neigh and Rs_ang
            body_descriptor_3 = tf.einsum('ijkl,ijl->ikl',Base_vector_ij_s, radial_ij) #shape=(nat,Ngauss_ang*self.nspec_embedding,ThetaN)

            body_descriptor_3 = tf.reshape(body_descriptor_3, [nat,-1])
            
            atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_3], axis=1)

            if self.body_order >= 4:
                body_tensor_4 = tf.einsum('ijkl,ijkl->ijkl',Base_vector_ij_s, Base_vector_ij_s)
                body_descriptor_4 = tf.einsum('ijkl,ijl->ikl',body_tensor_4, radial_ij)
                body_descriptor_4 = tf.reshape(body_descriptor_4, [nat, -1])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_4], axis=1)
            if self.body_order >= 5:
                body_tensor_5 = tf.einsum('ijkl,ijkl->ijkl',body_tensor_4, Base_vector_ij_s)
                body_descriptor_5 = tf.einsum('ijkl,ijl->ikl',body_tensor_5, radial_ij)
                body_descriptor_5 = tf.reshape(body_descriptor_5, [nat, -1])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_5], axis=1)
            


            #feature_size = Ngauss + Ngauss_ang * thetasN
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
        
        padding = tf.zeros((nmax_diff,3))
        forces = tf.concat([forces, padding], 0)
        #forces = tf.zeros((nmax_diff+nat,3))   
        
        return [tf.cast(total_energy, tf.float32), -tf.cast(forces, tf.float32), tf.cast(atomic_features, tf.float32)]
    
    def _get_atomic_numbers(self,x):
        atoms = x[0]
        nat = tf.cast(x[1], tf.int32)
        nmax = tf.cast(x[2], tf.int32)
        at_numbers = atoms.get_atomic_numbers()

        at_numbers = tf.pad(at_numbers, paddings=[[0,nmax-nat]]) 
        return at_numbers
    
    def _get_energy(self,atoms):
        return tf.cast(atoms.info['energy'], tf.float32)

    def _get_natom(self,atoms):
        return tf.cast(atoms.get_global_number_of_atoms(), tf.int32)
    def _get_positions(self,x):
        atoms = x[0]
        nat = tf.cast(x[1], tf.int32)
        nmax = tf.cast(x[2], tf.int32)
        positions = atoms.positions
        positions = tf.pad(positions, paddings=[[0,nmax-nat],[0,0]]) 
        return positions

    def _get_forces(self, x):
        atoms = x[0]
        nat = tf.cast(x[1], tf.int32)
        nmax = tf.cast(x[2], tf.int32)
        forces = atoms.get_forces()
        forces = tf.pad(forces, paddings=[[0,nmax-nat],[0,0]])
        return forces


    def call(self, inputs, training=False):
        '''inpu has a shape of batch_size x nmax_atoms x feature_size'''
        # may be just call the energy prediction here which will be needed in the train and test steps
        # the input are going to be filename from which descriptors and targets are going to be extracted

        #input is a batch of (nats, namx, ase atom object)

        batch_size = self.batch_size
        # the batch size may be different from the set batchsize saved in varaible self.batch_size
        # because the number of data point may not be exactly divisible by the self.batch_size.

        inputs_width = tf.ones(1)
        self.kn_rad = tf.reshape(self.rbf_nets(inputs_width[tf.newaxis, :]), [-1])
        self.kn_ang = tf.reshape(self.rbf_nets_ang(inputs_width[tf.newaxis, :]), [-1])
        #self.width_value = tf.reshape(self.width_nets(inputs_width[tf.newaxis, :]), [-1])

        #inputs_width_ang = tf.ones(1)
        #self.width_value_ang = tf.reshape(self.width_nets_ang(inputs_width_ang[tf.newaxis, :]), [-1])
        #inputs_zeta = tf.ones(1)
        #self.zeta_value = tf.reshape(self.zeta_nets(inputs_zeta[tf.newaxis, :]), [-1])
        self.zeta_value = self.zeta * tf.ones(1)
        #inputs for center networks
        tf_pi = tf.constant(math.pi, dtype=tf.float32)
        #Rs = tf.ones(1)
        #Rs_ang = tf.ones(1)
        theta_s = tf.ones(1)
        #self._Rs_rad = tf.reshape(self.Rs_rad_nets(Rs[tf.newaxis,:]), [-1]) * self.rcut
        #delta = (self.rcut - self.min_radial_center) / tf.cast(self.RsN_rad, tf.float32)
        #self._Rs_rad = help_fn.rescale_params(Rs_rad_pred, self.min_radial_center, self.rcut-delta)


        #self._Rs_ang = tf.reshape(self.Rs_ang_nets(Rs_ang[tf.newaxis,:]), [-1])*self.rcut_ang
        #delta = (self.rcut_ang - self.min_radial_center) / tf.cast(self.RsN_ang, tf.float32)
        #self._Rs_ang = help_fn.rescale_params(Rs_ang_pred, self.min_radial_center, self.rcut_ang-delta)
        
        self._thetas = tf.reshape(self.thetas_nets(theta_s[tf.newaxis,:]), [-1]) * tf_pi

        #self._thetas = help_fn.rescale_params(ts_pred, 0.0, tf_pi)


        batch_kn_rad = tf.tile([self.kn_rad], [batch_size,1])
        batch_kn_ang = tf.tile([self.kn_ang], [batch_size,1])

        #batch_width = tf.tile([self.width_value], [batch_size,1])
        #batch_width = tf.tile([self.width_value], [batch_size,1])
        #batch_width_ang = tf.tile([self.width_value_ang], [batch_size,1])
        batch_zeta = tf.tile([self.zeta_value], [batch_size,1])
        #batch_Rs_rad = tf.tile([self._Rs_rad], [batch_size,1])
        #batch_Rs_ang = tf.tile([self._Rs_ang], [batch_size,1])
        batch_theta_s = tf.tile([self._thetas], [batch_size,1])

        # todo: we need to pass nmax if we use padded tensors
        batch_nats = inputs[0]
        batch_nmax = inputs[1]
        nmax_diff = batch_nmax - batch_nats
        atoms = inputs[2]
        #positions and species_encoder are ragged tensors are converted to tensors before using them
        #positions = tf.reshape(inputs[0].to_tensor(shape=(-1,nmax,3)), (-1, 3*nmax))
        #positions = tf.reshape(inputs[0], (-1, 3*nmax))
        #positions = tf.cast(positions, dtype=tf.float32)
        #obtain species encoder

        spec_identity = tf.constant(self.species_identity, dtype=tf.int32) - 1
        species_one_hot_encoder = tf.one_hot(spec_identity, depth=self.nelement)
        self.trainable_species_encoder = self.species_nets(species_one_hot_encoder)

        species_encoder = tf.map_fn(self._get_atomic_numbers, (atoms,batch_nats,batch_nmax),
                               fn_output_signature=tf.float32,
                               parallel_iterations=self.batch_size)
        #species_encoder = inputs[1] #contains atomic number per atoms for all element in a batch
        batch_species_encoder = tf.zeros([batch_size, nmax, self.nspec_embedding], dtype=tf.float32)
        # This may be implemented better but not sure how yet
        for idx, spec in enumerate(self.species_identity):
            values = tf.ones([batch_size, nmax, self.nspec_embedding], dtype=tf.float32) * self.trainable_species_encoder[idx]
            batch_species_encoder += tf.where(tf.equal(tf.tile(species_encoder[:,:,tf.newaxis], [1,1,self.nspec_embedding]),
                                                       tf.cast(spec,tf.float32)),
                    values, tf.zeros([batch_size, nmax, self.nspec_embedding]))
        batch_species_encoder = tf.reshape(batch_species_encoder, [-1,self.nspec_embedding*nmax])
        #cells = tf.cast(inputs[3], dtype=tf.float32)
        #replica_idx = inputs[4]
        #C6 = tf.cast(inputs[5], dtype=tf.float32)

        elements = (batch_species_encoder, batch_kn_rad,
                nmax_diff, batch_nats,
                batch_zeta, batch_kn_ang,
                batch_theta_s, atoms)

        #elements = (batch_species_encoder, batch_width,
        #        positions, nmax_diff, batch_nats,
        #        batch_zeta, batch_width_ang, cells, replica_idx,
        #        batch_Rs_rad, batch_Rs_ang, batch_theta_s, C6)


        #energies, forces = tf.map_fn(self.tf_predict_energy_forces, elements,
        #energies, forces = self.compute_energy_forces(elements)    
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
        #inputs_target = data
        #inputs = inputs_target[:6]
        ###################################
        # data is understood to be a list of ase atom object
        #####################################3
        target = tf.map_fn(self._get_energy, data, 
                               fn_output_signature=tf.float32,
                               parallel_iterations=self.batch_size)
        target = tf.cast(target, tf.float32)

        batch_nats = tf.map_fn(self._get_natoms, data, 
                               fn_output_signature=tf.float32,
                               parallel_iterations=self.batch_size)

        batch_nats = tf.cast(batch_nats, tf.float32)

        nmax = tf.cast(tf.reduce_max(batch_nats), tf.float32)
        batch_nmax = tf.ones_like(batch_nats, dtype=tf.float32) * nmax

#        target_f = tf.reshape(inputs_target[7].to_tensor(), [-1, 3*nmax])
        target_f = tf.map_fn(self._get_forces, (data,batch_nats,batch_nmax),
                               fn_output_signature=tf.float32,
                               parallel_iterations=self.batch_size)

        #target_f = tf.reshape(inputs_target[7], [-1, 3*nmax])
        target_f = tf.cast(target_f, tf.float32)

        with tf.GradientTape() as tape:
            e_pred, forces, _ = self((batch_nats,batch_nmax,data), training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            ediff = (e_pred - target)
            forces = tf.reshape(forces, [-1, 3*nmax])

            #emse_loss = tf.reduce_mean((ediff/batch_nats)**2)
            emse_loss = tf.reduce_mean((ediff)**2)

            fmse_loss = tf.map_fn(help_fn.force_loss, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
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

        mae_f = tf.map_fn(help_fn.force_mae, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
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

         #   tf.summary.histogram('3. Parameters/1. width',self.width_value,self._train_counter)
            tf.summary.histogram('3. Parameters/1. kn_rad',self.kn_rad,self._train_counter)
            tf.summary.histogram('3. Parameters/2. kn_ang',self.kn_ang,self._train_counter)
          #  tf.summary.histogram('3. Parameters/2. width_ang',self.width_value_ang,self._train_counter)
            tf.summary.histogram('3. Parameters/3. zeta',self.zeta_value,self._train_counter)
          #  tf.summary.histogram('3. Parameters/4. Rs_rad',self._Rs_rad,self._train_counter)
          #  tf.summary.histogram('3. Parameters/5. Rs_ang',self._Rs_ang,self._train_counter)
            tf.summary.histogram('3. Parameters/6. thetas',self._thetas,self._train_counter)
        return {key: metrics[key] for key in metrics.keys()}


    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        #inputs_target = data
        #inputs = inputs_target[:6]
        #target = tf.cast(inputs_target[6], tf.float32)
        target = tf.map_fn(self._get_energy, data,
                               fn_output_signature=tf.float32,
                               parallel_iterations=self.batch_size)
        target = tf.cast(target, tf.float32)

        batch_nats = tf.map_fn(self._get_natoms, data,
                               fn_output_signature=tf.float32,
                               parallel_iterations=self.batch_size)

        batch_nats = tf.cast(batch_nats, tf.float32)

        nmax = tf.cast(tf.reduce_max(batch_nats), tf.float32)
        batch_nmax = tf.ones_like(batch_nats, dtype=tf.float32) * nmax

        target_f = tf.map_fn(self._get_forces, (data,batch_nats,batch_nmax),
                               fn_output_signature=tf.float32,
                               parallel_iterations=self.batch_size)

        #target_f = tf.reshape(inputs_target[7], [-1, 3*nmax])
        target_f = tf.cast(target_f, tf.float32)



        e_pred, forces, _ = self((batch_nats,batch_nmax,data), training=True)  # Forward pass

        forces = tf.reshape(forces, [-1, nmax*3])

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
        target = tf.map_fn(self._get_energy, data,
                               fn_output_signature=tf.float32,
                               parallel_iterations=self.batch_size)
        target = tf.cast(target, tf.float32)

        batch_nats = tf.map_fn(self._get_natoms, data,
                               fn_output_signature=tf.float32,
                               parallel_iterations=self.batch_size)

        batch_nats = tf.cast(batch_nats, tf.float32)

        nmax = tf.cast(tf.reduce_max(batch_nats), tf.float32)
        batch_nmax = tf.ones_like(batch_nats, dtype=tf.float32) * nmax

        target_f = tf.map_fn(self._get_forces, (data,batch_nats,batch_nmax),
                               fn_output_signature=tf.float32,
                               parallel_iterations=self.batch_size)

        #target_f = tf.reshape(inputs_target[7], [-1, 3*nmax])
        target_f = tf.cast(target_f, tf.float32)


        e_pred, forces, _ = self((batch_nats,batch_nmax,data), training=False)  # Forward pass
        forces = tf.reshape(forces, [-1, nmax*3])
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
