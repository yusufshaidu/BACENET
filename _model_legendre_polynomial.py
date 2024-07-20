from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
#import tensorflow
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
import mendeleev
from mendeleev import element
import math 
import itertools, os
from ase.io import read, write
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow_graphics as tfg
import tensorflow_graphics.math.spherical_harmonics as sph

def Networks(input_size, layer_sizes, 
             activations,
            weight_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_constraint=None,
            bias_constraint=None,
            prefix='main',
            l1=0.0,l2=0.0):

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    i = 0
    layer = 1
    for layer, activation in zip(layer_sizes[:-1], activations[:-1]):
        model.add(tf.keras.layers.Dense(layer, 
                                        activation=activation,
                                        kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        bias_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        activity_regularizer=None,
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        trainable=True,
                                        name=f'{prefix}_{i}_layer_{layer}_activation_{activation}'
                                        ))
        i += 1

    if activations[-1] == 'linear':
        model.add(tf.keras.layers.Dense(layer_sizes[-1],
                                        kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        trainable=True,
                                        name=f'{prefix}_{i}_layer_{layer}_activation_{activations[-1]}'
                                        ))
    else:
        model.add(tf.keras.layers.Dense(layer_sizes[-1], activation=activations[-1],
                                        kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
                                        activity_regularizer=None,
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        trainable=True,
                                        name=f'{prefix}_{i}_layer_{layer}_activation_{activations[-1]}'
                                        ))
        
    return model

class mBP_model(tf.keras.Model):
   
    def __init__(self, layer_sizes, rcut, species_identity, 
                 width, batch_size,
                activations,
                rc_ang,RsN_rad, RsN_ang,
                thetaN,width_ang,zeta,
                params_trainable=True,
                fcost=0.0,
                pbc=True,
                nelement=118,
                train_writer=None,
                l1=0.0,l2=0.0,
                nspec_embedding=64,
                train_zeta=True,
                include_vdw=False,
                rmin_u=3.0,
                rmax_u=5.0,
                rmin_d=11.0,
                rmax_d=12.0):
        
        #allows to use all the base class of tf.keras Model
        super().__init__()
        
        #self.loss_tracker = self.metrics.Mean(name='loss')
        self.layer_sizes = layer_sizes
        self.rcut = rcut
        self.species_identity = species_identity # atomic number
        self._width = width
        self.batch_size = batch_size
        self._params_trainable = params_trainable
        self._activations = activations
        self.rcut_ang = rc_ang
        self.RsN_rad = RsN_rad
        self.RsN_ang = RsN_ang
        self.thetaN = thetaN
        self.zeta = float(zeta)
        self.width_ang = float(width_ang)
        self.feature_size = self.RsN_rad + (self.RsN_ang * (self.thetaN+1))
        self.pbc = pbc
        self.fcost = float(fcost)
        self.nspecies = len(self.species_identity)
        self.train_writer = train_writer
        self.nspec_embedding = nspec_embedding
        self.train_zeta = train_zeta
        self.l1 = l1
        self.l2 = l2
        self.include_vdw = include_vdw
        self.rmin_u = rmin_u
        self.rmax_u = rmax_u
        self.rmin_d = rmin_d
        self.rmax_d = rmax_d
        
        # Layer is currectly noyt compactible with modelcheckpoints call back
        #self.width_nets = Linear(trainable=self._width_trainable)
        
        #self.width_value = self._width
        self.atomic_nets = Networks(self.feature_size, self.layer_sizes, self._activations, l1=self.l1, l2=self.l2)

        # the number of elements in the periodic table
        #self.nelement  = nelement
        self.nelement = nelement
        # create a species embedding network Nembedding x Nspecies
        self.species_nets = Networks(self.nelement, [self.nspec_embedding,1], ['sigmoid','linear'], prefix='species_encoder')

        if self._params_trainable:
            #constraint = tf.keras.constraints.MinMaxNorm(min_value=1e-2, 
            #                                             max_value=1.0, 
            #                                             rate=1.0, axis=0)
            #constraint = tf.keras.constraints.NonNeg()
            constraint = None

            self.width_nets = Networks(self.RsN_rad, [64,self.RsN_rad], ['sigmoid','softplus'],
                                      weight_initializer='ones',
                                      bias_initializer='ones',
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='radial_width')
            self.width_nets_ang = Networks(self.RsN_ang, [64,self.RsN_ang], ['sigmoid','softplus'],
                                      weight_initializer='ones',
                                      bias_initializer='ones',
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint,prefix='ang_width')
        self.lambda1_nets = Networks(self.thetaN, [64,self.thetaN], ['tanh','tanh'],
                                      weight_initializer='random_normal',
                                      bias_initializer='random_normal',
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='lambda1')
#        self.lambda2_nets = Networks(self.thetaN, [64,self.thetaN], ['tanh','tanh'],
#                                      weight_initializer='random_normal',
#                                      bias_initializer='random_normal',
#                                      kernel_constraint=constraint,
#                                      bias_constraint=constraint, prefix='lambda2_ijik')

    def switch(self, r, rmin,rmax):
        x = (r-rmin)/(rmax-rmin)
        res  = tf.zeros(tf.shape(x))
        res = tf.where(x<=0., 1.0, -6.0*x**5+15.0*x**4-10.0*x**3+1.0)
        res = tf.where(x>1.0, 0.0, res)
        return res

    def vdw_contribution(self,x):
        rmin_u = self.rmin_u
        rmax_u = self.rmax_u
        rmin_d = self.rmin_d
        rmax_d = self.rmax_d
        rij_norm = x[0]
        C6ij = x[1]

        rij_norm_inv6  = 1.0 / (rij_norm**6 + 1e-8)
        #rij_norm_inv2 = rij_norm_inv * rij_norm_inv
        #rij_norm_inv6 = rij_norm_inv2 * rij_norm_inv2 * rij_norm_inv2

        energy = -(1 - self.switch(rij_norm, rmin_u, rmax_u)) * self.switch(rij_norm,rmin_d, rmax_d) * rij_norm_inv6
        energy = energy * C6ij
        energy = 0.5 * tf.reduce_sum(energy)
        return [energy]

    

    def evaluate_single_LP(self,x):
        l = x[0]
        m = x[1]
        y = x[2]
        return tf.cast(sph.evaluate_legendre_polynomial([l],[m],y), tf.float32)

    def tf_legendre_p(self, x, l):
        
        lindx = tf.range(l+1)
        order_m = tf.range(l+1) * 0
        X = tf.tile([x], [l+1,1])
        p = tf.map_fn(self.evaluate_single_LP, (lindx,order_m,X), fn_output_signature=tf.float32)
        return p

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.float32)])
    def tf_fcut(self,r,rc):
        dim = tf.shape(r)
        pi = tf.constant(math.pi, dtype=tf.float32)
        return tf.where(tf.logical_and(r<=rc,r>1e-8), 0.5*(1.0 + tf.cos(pi*r/rc)), tf.zeros(dim, dtype=tf.float32))
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,None,None), dtype=tf.float32)])
    def tf_app_gaussian(self,x):
        # we approximate gaussians with polynomials (1+alpha x^2 / p)^(-p) ~ exp(-alpha x^2); 
        #p=64 is an even number

        p = 64.0

        args = tf.math.reciprocal(1.0 + x / p)
        args2 = args * args
        args4 = args2 * args2
        args8 = args4 * args4
        args16 = args8 * args8
        args32 = args16 * args16
        return args32 * args32


    @tf.function(
                input_signature=[(tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(3,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
                )])
    def tf_predict_energy_forces(self,x):

        rc = tf.cast(x[0],dtype=tf.float32)
        Ngauss = tf.cast(x[1], dtype=tf.int32)
        nmax_diff = tf.cast(x[5], dtype=tf.int32)
        nat = tf.cast(x[6], dtype=tf.int32)


        species_encoder = tf.cast(x[2][:nat], tf.float32)
        width = tf.cast(x[3], dtype=tf.float32)
        positions = tf.reshape(x[4][:nat*3], [nat,3])
        positions = tf.cast(positions, tf.float32)
        
        lambda1 = tf.cast(x[7], dtype=tf.float32)
        rc_ang = tf.cast(x[8], dtype=tf.float32)
        Ngauss_ang = tf.cast(x[9], dtype=tf.int32)
        thetasN = tf.cast(x[10], dtype=tf.int32)
        width_ang = tf.cast(x[11], dtype=tf.float32)

        #positions = tf.reshape(x[4])
        
        cell = tf.cast(x[12], tf.float32)
        replica_idx = tf.cast(x[13], tf.float32)
        evdw = 0.0
        if self.include_vdw:
            C6 = tf.cast(x[14], tf.float32)
        #lambda1 = tf.cast(x[14], tf.float32)

        
        
       # print(width) 
       # needs to be padded
        #file is a panna json for now
        r0 = tf.constant(0.25, dtype=tf.float32)
        
        #nat = len(positions)

        #Vectors = tf.zeros((nat, Ngauss), dtype=tf.float32)
        Rs = r0 + tf.range(Ngauss, dtype=tf.float32) * (rc-r0)/tf.cast(Ngauss, dtype=tf.float32)
        #eps = tf.constant(1e-3, dtype=tf.float32)
#        norm_ang = 1.0 + tf.sqrt(1.0 + eps)
        #adding the new tensor product increases the maximum
        # I am not sure what the appropriate normalization would be at the moment.


        tf_pi = tf.constant(math.pi, dtype=tf.float32)

        Rs_ang = r0 + tf.range(Ngauss_ang, dtype=tf.float32) * (rc_ang-r0)/tf.cast(Ngauss_ang, dtype=tf.float32)
        
        
        with tf.GradientTape() as g:
            g.watch(positions)
            
            if self.pbc:
                
                #l, m, n = replica_idx
                l = replica_idx[0]
                m = replica_idx[1]
                n = replica_idx[2]
               
                ll = tf.range(-l,l+1)
                mm = tf.range(-m,m+1)
                nn = tf.range(-n,n+1)
                
                tile_ll = tf.tile(ll[:,tf.newaxis,tf.newaxis], [1, tf.shape(mm)[0], tf.shape(nn)[0]])
                tile_ll = tf.expand_dims(tile_ll, 3) 
                tile_mm = tf.tile(mm[tf.newaxis,:,tf.newaxis], [tf.shape(ll)[0],1, tf.shape(nn)[0]]) 
                tile_mm = tf.expand_dims(tile_mm, 3) 
                tile_nn = tf.tile(nn[tf.newaxis,tf.newaxis,:], [tf.shape(ll)[0], tf.shape(mm)[0],1])
                tile_nn = tf.expand_dims(tile_nn, 3)
                
                replicas = tf.concat([tile_ll, tile_mm, tile_nn], axis=3) 
                replicas = tf.reshape(replicas,  [-1,3])

                #replicas = tf.constant(itertools.product(lmax,mmax,nmax), dtype=tf.float32)
                
                replicas = replicas @ cell
                #positions, replicas = make_replicas(positions, cell, rc, pbc=[True, True, True])
                positions_extended = tf.identity(positions)[:,tf.newaxis,:] + replicas
                nreplicas = tf.shape(replicas)
                n_replicas = nreplicas[0]
                positions_extended = tf.reshape(positions_extended, [nat*n_replicas, 3])
                
                species_encoder0 = tf.identity(species_encoder)

                species_encoder = tf.reshape(tf.tile([species_encoder], [1, n_replicas]), [nat*n_replicas])
                if self.include_vdw:
                    C6_extended = tf.reshape(tf.tile([C6], [1, n_replicas]), [nat*n_replicas])
            else:
                positions_extended = tf.identity(positions)
                species_encoder0 = tf.identity(species_encoder)
                if self.include_vdw:
                    C6_extended = tf.identity(C6)
            #positions_extended has nat x nreplica x 3 for periodic systems and nat x 1 x 3 for molecules
            
            #currectly use all neighbors to compute the radial part of the descriptors
            #rj-ri
            #Nneigh x nat x 3
            all_rij = positions - positions_extended[:, tf.newaxis, :]
            all_rij = tf.transpose(all_rij, [1,0,2])
            
            #all_rij_norm = tf.linalg.norm(all_rij, axis=-1)
            #nat, Nneigh: Nneigh = nat * n_replicas
            #regularize the norm to avoid divition by zero in the derivatives
            all_rij_norm = tf.sqrt(tf.reduce_sum(all_rij * all_rij, axis=-1) + 1e-16)
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

            inball_rad = tf.where(tf.logical_and(all_rij_norm <=rc, all_rij_norm>1e-8))
            all_rij_norm = tf.gather_nd(all_rij_norm, inball_rad)
            #produces as list of tensors with different shapes because atoms have different number of neighbors
            all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm, inball_rad[:,0]).to_tensor(default_value=1e-8)
            
            all_rij = tf.gather_nd(all_rij, inball_rad)
            #produces as list of tensors with different shapes because atoms have different number of neighbors
            all_rij = tf.RaggedTensor.from_value_rowids(all_rij, inball_rad[:,0]).to_tensor(default_value=1e-8)
            
            species_encoder_rad = tf.tile([species_encoder], [nat, 1])

            species_encoder_rad = tf.gather_nd(species_encoder_rad, inball_rad)
            species_encoder_rad = tf.RaggedTensor.from_value_rowids(species_encoder_rad, inball_rad[:,0]).to_tensor()

            
            
            #nat x Ngauss x Nneigh
            gauss_args = width[tf.newaxis,:,tf.newaxis] * (all_rij_norm[:,tf.newaxis,:] - Rs[tf.newaxis,:,tf.newaxis])**2
            #mask the non-zero values after ragged tensor is converted to fixed shape
            # This is very important to avoid adding contributions from atoms outside the sphere
            #nonzero_val = tf.where(all_rij_norm > 1e-8, 1.0,0.0)
            #nonzero_val = tf.cast(nonzero_val, tf.float32)
            #gauss_args = gauss_args * nonzero_val[:,tf.newaxis,:]



            #gauss_args = tf.einsum('j, ijk->ijk', width, (all_rij_norm[:,tf.newaxis,:] - Rs[tf.newaxis,:,tf.newaxis])**2)


            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x 1 x Nneigh
            #fcuts is nat x Nneigh and reshaped to nat x 1 x Nneigh
            args = species_encoder_rad[:,tf.newaxis,:] * self.tf_app_gaussian(gauss_args) * self.tf_fcut(all_rij_norm, rc)[:,tf.newaxis,:]
            
            # sum over neighbors j including periodic boundary conditions
            atomic_descriptors = tf.reduce_sum(args, axis=-1)



            #implement angular part: compute vect_rij dot vect_rik / rij / rik
                   
            inball_ang = tf.where(all_rij_norm <=rc_ang)
            
            #idx = tf.where(tf.logical_and(a<6, a>0))
            all_rij_norm = tf.gather_nd(all_rij_norm, inball_ang)
            #produces a list of tensors with different shapes because atoms have different number of neighbors

            all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm, inball_ang[:,0]).to_tensor(default_value=1e-8)
            
            all_rij = tf.gather_nd(all_rij, inball_ang)
            #produces as list of tensors with different shapes because atoms have different number of neighbors
            all_rij = tf.RaggedTensor.from_value_rowids(all_rij, inball_ang[:,0]).to_tensor(default_value=1e-8)
            
            species_encoder = tf.tile([species_encoder], [nat, 1]) 

            species_encoder = tf.gather_nd(species_encoder, inball_ang)
            species_encoder = tf.RaggedTensor.from_value_rowids(species_encoder, inball_ang[:,0]).to_tensor()

            reg = 1e-20
            all_rij_norm_inv = 1.0 / (all_rij_norm + reg)
            
            #number of neighbors:
            _Nneigh = tf.shape(all_rij)
            Nneigh = _Nneigh[1]
            
            rij_unit = all_rij * all_rij_norm_inv[:,:,tf.newaxis]
            #nat x Neigh X Nneigh
            #cos_theta_ijk = tf.matmul(rij_unit, tf.transpose(rij_unit, (0,2,1)))
            #rij.rik
            cos_theta_ijk = tf.einsum('ijk, ilk -> ijl', rij_unit, rij_unit) 

            #do we need to remove the case of j=k?
            #i==j or i==k already contribute 0 because of 1/2**zeta or  constant
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
            cos_theta_ijk = tf.reshape(cos_theta_ijk, [nat,-1])
            reg2 = 1e-6
            cos_theta_ijk = tf.clip_by_value(cos_theta_ijk, clip_value_min=-1.0+reg2, clip_value_max=1.0-reg2)
            #compute the polinomials up to max l: currently lmax = thetasN
            legendre_polynomials = self.tf_legendre_p(tf.reshape(cos_theta_ijk. [-1]), thetasN)
            #legendre_polynomials is of shape nat*Neigh*Neigh*thetasN
            legendre_polynomials = tf.reshape(legendre_polynomials, [nat,-1,thetasN+1])


            legendre_polynomials_weights = legendre_polynomials * lambda1[tf.newaxis,tf.newaxis,:] 

            #(rij + rik) / 2
            #dim : nat,neigh,neigh
            rij_p_rik = all_rij_norm[:,tf.newaxis,:] + all_rij_norm[:,:,tf.newaxis]
            #Nat x Nneigh**2
            rij_p_rik = tf.reshape(rij_p_rik, [nat, -1])
            #Nat x Ngauss_ang * Nneigh**2
            rij_p_rik_rs = width_ang[tf.newaxis,tf.newaxis,:] * (rij_p_rik[:,:,tf.newaxis]/2.0  - Rs_ang[tf.newaxis,tf.newaxis,:])**2
            #mask the non-zero values after ragged tensor is converted to fixed shape
            #nonzero_val = tf.where(rij_p_rik > 1e-8, 1.0,0.0)
            #nonzero_val = tf.cast(nonzero_val, tf.float32)
            #rij_p_rik_rs = rij_p_rik_rs * nonzero_val[:,tf.newaxis,:]


            #compute the exponetial term
            exp_ang_term = self.tf_app_gaussian(rij_p_rik_rs)
            
            exp_ang_theta_ijk = legendre_polynomials_weights[:,:,:,tf.newaxis] * exp_ang_term[:,:,tf.newaxis,:]
            exp_ang_theta_ijk = tf.reshape(exp_ang_theta_ijk, [nat,-1, (thetasN+1)*Ngauss_ang])
            
            #exp_ang_tensor_ijk = tensors_contrib[:,tf.newaxis,:,:] * exp_ang_term[:,:,tf.newaxis,:]
            #exp_ang_tensor_ijk = tf.reshape(exp_ang_tensor_ijk, [nat,thetasN*Ngauss_ang,-1])

            #nat x Nneigh x Nneigh
            fc_rij_rik = self.tf_fcut(all_rij_norm, rc_ang)[:,tf.newaxis,:] * self.tf_fcut(all_rij_norm, rc_ang)[:,:,tf.newaxis]
            #fc_rij_rik = tf.gather_nd(fc_rij_rik_all, cond)
            #fc_rij_rik = tf.reshape(tf.RaggedTensor.from_value_rowids(fc_rij_rik, cond[:,0]).to_tensor(), [nat,-1])
            
            #convert species_encoder from nat,Nneig=nat*n_replicas to nat,Nneig x Nneig to have i,j,k
            species_encoder_jk = species_encoder[:,tf.newaxis,:] * species_encoder[:,:,tf.newaxis]

            #collect all terms
            spec_encoder_fcjk = species_encoder_jk * fc_rij_rik
            #nat x Neigh**2
            spec_encoder_fcjk = tf.reshape(spec_encoder_fcjk, [nat, -1])
            _descriptor_ang = spec_encoder_fcjk[:,:,tf.newaxis] * exp_ang_theta_ijk
            descriptor_ang = tf.reduce_sum(_descriptor_ang, axis=1)

            #_descriptor_tensor = spec_encoder_fcjk[:,tf.newaxis,:] * exp_ang_tensor_ijk
            #descriptor_tensor = tf.reduce_sum(_descriptor_tensor, axis=-1)

            #atomic_descriptors = tf.concat([atomic_descriptors, descriptor_ang, descriptor_tensor], axis=1)
            atomic_descriptors = tf.concat([atomic_descriptors, descriptor_ang], axis=1)
            
            #feature_size = Ngauss + Ngauss_ang * thetasN
            #the descriptors can be scaled
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])
            
            #mutiply by species weights per atom
            atomic_descriptors = species_encoder0[:,tf.newaxis] * atomic_descriptors
            #predict energy and forces

            atomic_energies = self.atomic_nets(atomic_descriptors)
            total_energy = tf.reduce_sum(atomic_energies)

            if self.include_vdw:
                total_energy += evdw

           # print(total_energy)
       # print(positions, positions)   
        forces = g.gradient(total_energy, positions)
        
        #tf.debugging.check_numerics(total_energy, message='Total_energy contains NaN')
        #tf.debugging.check_numerics(positions, message='positions contains NaN')
        #tf.debugging.check_numerics(forces, message='forces contains NaN')
        padding = tf.zeros((nmax_diff,3))
        forces = tf.concat([forces, padding], 0)
        #forces = tf.zeros((nmax_diff+nat,3))   
        return [tf.cast(total_energy, tf.float32), -tf.cast(forces, tf.float32)]
    def call(self, inputs, training=False):
        '''inpu has a shape of batch_size x nmax_atoms x feature_size'''
        # may be just call the energy prediction here which will be needed in the train and test steps
        # the input are going to be filename from which descriptors and targets are going to be extracted

        batch_size = tf.shape(inputs[2])[0]

        rcuts = tf.tile([self.rcut], [batch_size])
        rcuts_ang = tf.tile([self.rcut_ang], [batch_size])
        batch_RsN_rad = tf.tile([self.RsN_rad], [batch_size])
        batch_RsN_ang = tf.tile([self.RsN_ang], [batch_size])

        batch_thetaN = tf.tile([self.thetaN], [batch_size])

        if self._params_trainable:
            #inputs_width = tf.ones(self.RsN_rad) * self._width
            #exponentially samples gaussian width
            m = tf.range(1,self.RsN_rad+1, dtype=tf.float32)
            N = tf.cast(self.RsN_rad, tf.float32)
            inputs_width = (N**(m/N) / self.rcut)**2
            self.width_value = tf.reshape(self.width_nets(inputs_width[tf.newaxis, :]), [-1])

            m = tf.range(1,self.RsN_ang+1, dtype=tf.float32)
            N = tf.cast(self.RsN_ang, tf.float32)
            inputs_width_ang = (N**(m/N) / self.rcut_ang)**2
            #inputs_width_ang = tf.ones(self.RsN_ang) * self.width_ang
            self.width_value_ang = tf.reshape(self.width_nets_ang(inputs_width_ang[tf.newaxis, :]), [-1])
            #self.width_value_ang += self.width_ang
            _in_lambda = tf.ones(self.thetaN+1)
            self.lambda1 = tf.reshape(self.lambda1_nets(_in_lambda[tf.newaxis, :]), [-1])
        else:
            self.width_value = tf.ones(self.RsN_rad) * self._width
            self.width_value_ang = tf.ones(self.RsN_ang) * self.width_ang
            self.lambda1 = tf.ones(self.thetaN+1)

        batch_width = tf.tile([self.width_value], [batch_size,1])
        batch_width_ang = tf.tile([self.width_value_ang], [batch_size,1])
        batch_lambda = tf.tile([self.lambda1], [batch_size,1])

        #lambda_init = tf.ones(self.thetaN)[tf.newaxis,:]
        #self.lambda1 = tf.reshape(self.lambda1_nets(lambda_init), [-1])
        #batch_lambda1 = tf.tile([self.lambda1], [batch_size,1])

#        lambda2 = tf.reshape(self.lambda2_nets(lambda_init), [-1])
#        batch_lambda2 = tf.tile([lambda2], [batch_size,1])

        #positions = tf.reshape(inputs[0], (self.batch_size, -1))
        #species_encoder = tf.reshape(inputs[1], (self.batch_size, -1))

        batch_nats = inputs[2]
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        batch_nmax = tf.tile([nmax], [batch_size])
        #print(batch_nats, batch_nmax)
        nmax_diff = batch_nmax - batch_nats

        #positions and species_encoder are ragged tensors are converted to tensors before using them
        positions = tf.reshape(inputs[0].to_tensor(shape=(-1,nmax,3)), (-1, 3*nmax))
        #obtain species encoder

        # if we want species encoder for every elements, then we can do this
        # I am not sure it is useful to encode all species except in other context.
        spec_identity = tf.constant(self.species_identity, dtype=tf.int32) - 1

        #spec_identity = tf.range(len(self.species_identity), dtype=tf.int32)

        species_one_hot_encoder = tf.one_hot(spec_identity, depth=self.nelement)

        trainable_species_encoder = self.species_nets(species_one_hot_encoder)
        species_encoder = inputs[1].to_tensor(shape=(-1, nmax))
        batch_species_encoder = tf.zeros([batch_size, nmax], dtype=tf.float32)

        for idx, spec in enumerate(self.species_identity):
            values = tf.ones([batch_size, nmax], dtype=tf.float32) * trainable_species_encoder[idx]
            batch_species_encoder += tf.where(tf.equal(species_encoder,tf.cast(spec,tf.float32)),
                    values, tf.zeros([batch_size, nmax]))


        cells = inputs[3]
        replica_idx = inputs[4]
        C6 = inputs[5]

        elements = (rcuts, batch_RsN_rad, batch_species_encoder, batch_width,
                positions, nmax_diff, batch_nats,
                batch_lambda, rcuts_ang, batch_RsN_ang,
                batch_thetaN, batch_width_ang, cells, replica_idx, C6)

        energies, forces = tf.map_fn(self.tf_predict_energy_forces, elements, fn_output_signature=[tf.float32, tf.float32])
        return energies, forces

    def force_loss(self, x):

        nat = tf.cast(x[0], tf.int32)
        force_ref = tf.reshape(x[1][:3*nat], (nat,3))
        force_pred = tf.reshape(x[2][:3*nat], (nat,3))

        loss = tf.reduce_mean((force_ref - force_pred)**2)
        #loss = tf.reduce_sum((force_ref - force_pred)**2)

        return loss

    def force_mse(self, x):

        nat = tf.cast(x[0], tf.int32)
        force_ref = tf.reshape(x[1][:3*nat], (nat,3))
        force_pred = tf.reshape(x[2][:3*nat], (nat,3))

        fmse = tf.reduce_mean((force_ref - force_pred)**2)

        return fmse

    def force_mae(self, x):

        nat = tf.cast(x[0], tf.int32)
        force_ref = tf.reshape(x[1][:3*nat], (nat,3))
        force_pred = tf.reshape(x[2][:3*nat], (nat,3))

        fmae = tf.reduce_mean(tf.abs(force_ref - force_pred))

        return fmae

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs_target = data
        inputs = inputs_target[:6]
        target = inputs_target[6]

        #in_shape = inputs[0].shape

        # the last dimension is the feature size
        #inputs = tf.reshape(inputs, [-1, in_shape[-1]])

        batch_nats = tf.cast(inputs[2], tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)

        target_f = tf.reshape(inputs_target[7].to_tensor(), [-1, 3*nmax])
        target_f = tf.cast(target_f, tf.float32)

        with tf.GradientTape() as tape:
            e_pred, forces = self(inputs, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            ediff = (e_pred - target)
            forces = tf.reshape(forces, [-1, 3*nmax])

            #emse_loss = tf.reduce_mean((ediff/batch_nats)**2)
            emse_loss = tf.reduce_mean((ediff)**2)

            fmse_loss = tf.map_fn(self.force_loss, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
            fmse_loss = tf.reduce_mean(fmse_loss)
            #dforces = tf.reshape(target_f, [self.batch_size, 3*nmax]) - tf.reshape(forces, [self.batch_size, 3*nmax])
            #fmse_loss = tf.reduce_mean(tf.reduce_sum((dforces)**2, axis=-1) / (3*batch_nats))

            #loss = self.compute_loss(y=target, y_pred=e_pred)
            #loss += self.fcost * self.compute_loss(y=target_f, y_pred=forces)
            loss = emse_loss
            loss += self.fcost * fmse_loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        #trainable_vars.append(self.width)
        #print(trainable_vars)
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))



        # Update metrics (includes the metric that tracks the loss)
#        ediff = (e_pred - target)

        metrics = {'tot_st': self._train_counter}
        mae = tf.reduce_mean(tf.abs(ediff / batch_nats))
        rmse = tf.sqrt(tf.reduce_mean((ediff/batch_nats)**2))

 #       dforces = tf.reshape(target_f, [self.batch_size, 3*nmax]) - tf.reshape(forces, [self.batch_size, 3*nmax])
        #dforces = dforces  / (3*batch_nats[:, tf.newaxis])

        mae_f = tf.map_fn(self.force_mae, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        mae_f = tf.reduce_mean(mae_f)

        rmse_f = tf.sqrt(fmse_loss)


        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})

        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})
#        metrics.update({'Parameter-width-rad':width})
 #       metrics.update({'Parameter-width-ang':width_ang})
 #       metrics.update({'Parameter-zeta':zeta})

        #    if self._f_loss:
        #        metrics.update({'F_MAE': Fmae})
        #if 'RMSE' in self._printmetrics:
        #    metrics.update({'RMSE/at': rmseat})
        #    if self._f_loss:
        #        metrics.update({'F_RMSE': Frmse})

        metrics.update({'loss': loss})
        metrics.update({'energy loss': emse_loss})
        metrics.update({'force loss': fmse_loss})
        lr = K.eval(self.optimizer.lr)

        #with writer.set_as_default():
        with self.train_writer.as_default(step=self._train_counter):

            tf.summary.scalar('1. Losses/1. Total',loss,self._train_counter)
            tf.summary.scalar('2. Metrics/1. RMSE/atom',rmse,self._train_counter)
            tf.summary.scalar('2. Metrics/2. MAE/atom',mae,self._train_counter)
            tf.summary.scalar('2. Metrics/3. RMSE_F',rmse_f,self._train_counter)
            tf.summary.scalar('2. Metrics/3. MAE_F',mae_f,self._train_counter)
            tf.summary.scalar('4. LearningRate/1. LR',lr,self._train_counter)
            tf.summary.histogram('3. Parameters/1. width',self.width_value,self._train_counter)
            tf.summary.histogram('3. Parameters/2. width_ang',self.width_value_ang,self._train_counter)
            tf.summary.histogram('3. Parameters/3. zeta',self.zeta_value,self._train_counter)
#            tf.summary.histogram('3. Parameters/4. lambda',self.lambda1,self._train_counter)

        #self.metrics.update(metrics)

        #for metric in self.metrics:
        #    if metric.name == "loss":
        #        metric.update_state(loss)
        #    else:
        #        metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value

        #return {m.name: m.result() for m in self.metrics}
        return {key: metrics[key] for key in metrics.keys()}


    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs_target = data
        inputs = inputs_target[:6]
        target = inputs_target[6]

        #in_shape = inputs[0].shape
        e_pred, forces = self(inputs, training=True)  # Forward pass

        # Update metrics (includes the metric that tracks the loss)

        batch_nats = tf.cast(inputs[2], tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)

        forces = tf.reshape(forces, [-1, nmax*3])

        target_f = tf.reshape(inputs_target[7].to_tensor(), [-1, 3*nmax])
        target_f = tf.cast(target_f, tf.float32)

        ediff = (e_pred - target)

        mae = tf.reduce_mean(tf.abs(ediff / batch_nats))
        rmse = tf.sqrt(tf.reduce_mean((ediff/batch_nats)**2))

        fmse_loss = tf.map_fn(self.force_loss, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        fmse_loss = tf.reduce_mean(fmse_loss)

        #dforces = tf.reshape(target_f, [self.batch_size, 3*nmax]) - tf.reshape(forces, [self.batch_size, 3*nmax])
        mae_f = tf.map_fn(self.force_mae, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        mae_f = tf.reduce_mean(mae_f)
        rmse_f = tf.sqrt(fmse_loss)

        metrics = {}
        #mae = tf.reduce_mean(tf.abs(ediff))
        loss = rmse * rmse + self.fcost * fmse_loss
        #rmse = tf.sqrt(loss)

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
        inputs_target = data
        inputs = inputs_target[:6]
        target = inputs_target[6]
                #in_shape = inputs[0].shape
        e_pred, forces = self(inputs, training=False)  # Forward pass

        batch_nats = tf.cast(inputs[2], tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)

        forces = tf.reshape(forces, [-1, nmax*3])

        target_f = tf.reshape(inputs_target[7].to_tensor(), [-1, 3*nmax])
        target_f = tf.cast(target_f, tf.float32)

        ediff = (e_pred - target)


        mae = tf.reduce_mean(tf.abs(ediff / batch_nats))
        rmse = tf.sqrt(tf.reduce_mean((ediff/batch_nats)**2))

        fmse = tf.map_fn(self.force_mse, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        fmse = tf.reduce_mean(fmse)

        #dforces = tf.reshape(target_f, [self.batch_size, 3*nmax]) - tf.reshape(forces, [self.batch_size, 3*nmax])
        mae_f = tf.map_fn(self.force_mae, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        mae_f = tf.reduce_mean(mae_f)

        rmse_f = tf.sqrt(fmse)

        metrics = {}
        #mae = tf.reduce_mean(tf.abs(ediff))
        #loss = tf.reduce_mean(ediff**2)
        #rmse = tf.sqrt(loss)

        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})
        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})

        return [target, e_pred, metrics, tf.reshape(target_f, [-1, nmax, 3]),tf.reshape(forces, [-1, nmax, 3]), batch_nats]
