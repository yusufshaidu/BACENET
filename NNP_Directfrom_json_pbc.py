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

import argparse
from multiprocessing import Pool
from functools import partial


class Linear(tf.keras.layers.Layer):
    def __init__(self, trainable=True):
        super().__init__()
        self._trainable = trainable
        self.w = self.add_weight(
            shape=(), initializer="random_normal", trainable=self._trainable)
        self.b = self.add_weight(shape=(), initializer="random_normal", trainable=self._trainable)

    def call(self, inputs):
        #print(self.w, self.b)
        if self._trainable:
            return tf.nn.relu(inputs * self.w + self.b)
        return inputs



def Networks(input_size, layer_sizes, 
             activations,
            weight_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_constraint=None,
            bias_constraint=None,
            prefix='main'):

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    i = 0
    layer = 1
    for layer, activation in zip(layer_sizes[:-1], activations[:-1]):
        model.add(tf.keras.layers.Dense(layer, 
                                        activation=activation,
                                        kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=None,
                                        bias_regularizer=None,
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
                params_trainable,
                activations,
                rc_ang,RsN_rad, RsN_ang,
                thetaN,width_ang,zeta,
                order=2,
                fcost=0.0,
                pbc=True):
        
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
        self.order = order
        self.feature_size = self.RsN_rad + self.RsN_ang * self.thetaN
        self.pbc = pbc
        self.fcost = float(fcost)
        self.nspecies = len(self.species_identity)

              
        
        # Layer is currectly noyt compactible with modelcheckpoints call back
        #self.width_nets = Linear(trainable=self._width_trainable)
        
        #self.width_value = self._width
        self.atomic_nets = Networks(self.feature_size, self.layer_sizes, self._activations)

        # the number of elements in the periodic table
        nelement  = 118
        self.nelement = nelement
        # create a species embedding network Nembedding x Nspecies
        self.species_nets = Networks(nelement, [16,1], ['sigmoid','linear'], prefix='species_encoder')

        if self._params_trainable:
            #constraint = tf.keras.constraints.MinMaxNorm(min_value=1e-2, 
            #                                             max_value=1.0, 
            #                                             rate=1.0, axis=0)
            #constraint = tf.keras.constraints.NonNeg()
            constraint = None

            self.width_nets = Networks(1, [1], ['linear'],
                                      weight_initializer='random_normal',
                                      bias_initializer='random_normal',
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='radial_width')
            self.width_nets_ang = Networks(1, [1], ['linear'],
                                      weight_initializer='random_normal',
                                      bias_initializer='random_normal',
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint,prefix='ang_width')
            
            #constraint = tf.keras.constraints.MinMaxNorm(min_value=0.2, 
            #                                             max_value=1.0, 
            #                                             rate=0.75, axis=0)
            self.zeta_nets = Networks(1, [1], ['linear'],
                                      weight_initializer='random_normal',
                                      bias_initializer='random_normal',
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='zeta')
       # self.atomic_nets = LayerNets(self.layer_sizes, self.feature_size)
        
        #self.inputs = tf.keras.Input(shape=(self.feature_size,))
        
        #print("I am here")
        
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.float32)])
    def tf_fcut(self,r,rc):
        dim = tf.shape(r)
        pi = tf.constant(math.pi, dtype=tf.float32)
        return tf.where(r<=rc, 0.5*(1.0 + tf.cos(pi*r/rc)), tf.zeros(dim, dtype=tf.float32))
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,None,None), dtype=tf.float32)])
    def tf_app_gaussian(self,x):
        # we approximate gaussians with polynomials (1+alpha x^2 / p)^(-p) ~ exp(-alpha x^2); 
        #p=64 is an even number

        p = 64.0

        args = tf.math.reciprocal(1.0 + x / p + 1e-8)
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
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(3,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32)
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
        
        zeta = tf.cast(x[7], dtype=tf.float32)
        rc_ang = tf.cast(x[8], dtype=tf.float32)
        Ngauss_ang = tf.cast(x[9], dtype=tf.int32)
        thetasN = tf.cast(x[10], dtype=tf.int32)
        width_ang = tf.cast(x[11], dtype=tf.float32)

        #positions = tf.reshape(x[4])
        
        cell = tf.cast(x[12], tf.float32)
        replica_idx = tf.cast(x[13], tf.float32)
        
        
       # print(width) 
       # needs to be padded
        #file is a panna json for now
        r0 = tf.constant(0.25, dtype=tf.float32)
        
        #nat = len(positions)

        #Vectors = tf.zeros((nat, Ngauss), dtype=tf.float32)
        Rs = r0 + tf.range(Ngauss, dtype=tf.float32) * (rc-r0)/tf.cast(Ngauss, dtype=tf.float32)
        eps = tf.constant(1e-3, dtype=tf.float32)
        norm_ang = 1.0 + tf.sqrt(1.0 + eps)
        norm_ang = 2.0
        tf_pi = tf.constant(math.pi, dtype=tf.float32)

        Rs_ang = r0 + tf.range(Ngauss_ang, dtype=tf.float32) * (rc_ang-r0)/tf.cast(Ngauss_ang, dtype=tf.float32)
        theta_s = tf.range(thetasN, dtype=tf.float32) * tf_pi / tf.cast(thetasN, dtype=tf.float32)

        sin_theta_s = tf.sin(theta_s)
        cos_theta_s = tf.cos(theta_s)
        sin_theta_s2 = sin_theta_s * sin_theta_s
        
        
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
            else:
                positions_extended = tf.identity(positions)
                #species_encoder = species_encoder[:,tf.newaxis]
            #positions_extended has nat x nreplica x 3 for periodic systems and nat x 1 x 3 for molecules
            
            #rj-ri
            #Nneigh x nat x 3
            all_rij = positions - positions_extended[:, tf.newaxis, :]
            all_rij = tf.transpose(all_rij, [1,0,2])
            #all_rij_norm = tf.linalg.norm(all_rij, axis=-1)
            #nat, Nneigh: Nneigh = nat * n_replicas
            #regularize the norm to avoid divition by zero in the derivatives
            all_rij_norm = tf.sqrt(tf.reduce_sum(all_rij * all_rij, axis=-1) + 1e-12)
            
            #inball_rad = tf.where(tf.logical_and(all_rij_norm <=rc, all_rij_norm>1e-8))
            
            #idx = tf.where(tf.logical_and(a<6, a>0))
            #all_rij_norm = tf.gather_nd(all_rij_norm, inball_rad)
            #produces as list of tensors with different shapes because atoms have different number of neighbors
            #all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm, inball_rad[:,0])
            
            #all_rij = tf.gather_nd(all_rij, inball_rad)
            #produces as list of tensors with different shapes because atoms have different number of neighbors
            #all_rij= tf.RaggedTensor.from_value_rowids(all_rij, inball_rad[:,0])
            
            #species_encoder = tf.gather_nd(tf.tile([species_encoder],[1,nat], inball_rad)
            
            #print(all_rij.to_tensor())
            
            #all_rij_norm = tf.where(all_rij_norm <= rc, all_rij_norm, tf.zeros((nat,nat)))
            #all_rij_norm += 1e-10
            #all_rij = tf.where(tf.tile(all_rij_norm[:,:,tf.newaxis],[1,1,3]) <= rc, all_rij, tf.zeros((nat,nat,3)))
            #all_rij += 1e-10

            #all_rij = tf.gather_ng(all_rij, idx_in_ball)
            #all_rij_norm = tf.gather_ng(all_rij_norm, idx_in_ball)

            #nat x Ngauss x Nneigh
            gauss_args = width * (all_rij_norm[:,tf.newaxis,:] - Rs[tf.newaxis,:,tf.newaxis])**2

            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to 1 x 1 x Nneigh
            #fcuts is nat x Nneigh and reshaped to nat x 1 x Nneigh
            args = species_encoder[tf.newaxis,tf.newaxis,:] * self.tf_app_gaussian(gauss_args) * self.tf_fcut(all_rij_norm, rc)[:,tf.newaxis,:]
            
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
            
#            tf.debugging.check_numerics(all_rij_norm_inv, message='inverse rij contains NaN')
            
            rij_unit = all_rij * all_rij_norm_inv[:,:,tf.newaxis]
            #nat x Neigh X Nneigh
            cos_theta_ijk = tf.matmul(rij_unit, tf.transpose(rij_unit, (0,2,1)))
        
            
            #tf.debugging.check_numerics(rij_dot_rik, message='dot product of rij contains NaN')
            
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
            
            #cos_theta_ijk = tf.reshape(tf.RaggedTensor.from_value_rowids(cos_theta_ijk, cond[:,0]).to_tensor(default_value=1e-8), [nat,-1])
            reg2 = 1e-6
            cos_theta_ijk = tf.clip_by_value(cos_theta_ijk, clip_value_min=-1+reg2, clip_value_max=1.0-reg2)

            cos_theta_ijk2 = cos_theta_ijk * cos_theta_ijk
            #nat x thetasN x N_unique
            #sin_theta_ijk = tf.sqrt(1.0 - cos_theta_ijk2[:,tf.newaxis,:] + eps * sin_theta_s2[tf.newaxis,:,tf.newaxis])
            sin_theta_ijk = tf.sqrt(1.0 - cos_theta_ijk2)
           # tf.debugging.check_numerics(sin_theta_ijk, message='sin theta contains NaN')
            cos_theta_ijk_theta_s = cos_theta_ijk[:,tf.newaxis,:] * cos_theta_s[tf.newaxis,:,tf.newaxis] 
            #tf.debugging.check_numerics(cos_theta_ijk_theta_s, message='cos(theta - theta_s) without sin term contains NaN')
            
            cos_theta_ijk_theta_s += (sin_theta_ijk[:,tf.newaxis,:] * sin_theta_s[tf.newaxis,:,tf.newaxis])
            #tf.debugging.check_numerics(cos_theta_ijk_theta_s, message='cos(theta - theta_s) with sin term contains NaN')
            #compute 1+cos(theta_ijk - theta_s)
            cos_theta_ijk_theta_s += 1.0
            cos_theta_ijk_theta_s_zeta = (cos_theta_ijk_theta_s / norm_ang)**zeta
           # tf.debugging.check_numerics(cos_theta_ijk_theta_s_zeta, message='cos(theta - theta_s) contains NaN')

            #cos_theta_ijk_theta_s = tf.transpose(cos_theta_ijk_theta_s, (1,2,0))

            #(rij + rik) / 2
            #dim : nat,nat,nat
            rij_p_rik = all_rij_norm[:,tf.newaxis,:] + all_rij_norm[:,:,tf.newaxis]
            #rij_p_rik = tf.gather_nd(rij_p_rik_all, cond)
            #rij_p_rik = tf.reshape(tf.RaggedTensor.from_value_rowids(rij_p_rik, cond[:,0]).to_tensor(), [nat,-1])
            #Nat x Nneigh**2
            rij_p_rik = tf.reshape(rij_p_rik, [nat, -1])
            #Nat x Ngauss_ang * Nneigh**2
            rij_p_rik_rs = width_ang * (rij_p_rik[:,tf.newaxis,:]/2.0  - Rs_ang[tf.newaxis,:,tf.newaxis])**2
            #compute the exponetial term
            exp_ang_term = self.tf_app_gaussian(rij_p_rik_rs)
 #           tf.debugging.check_numerics(exp_ang_term, message='angular gaussian contains NaN')
            
            exp_ang_theta_ijk = cos_theta_ijk_theta_s_zeta[:,tf.newaxis,:,:] * exp_ang_term[:,:,tf.newaxis,:]
            
            exp_ang_theta_ijk = tf.reshape(exp_ang_theta_ijk, [nat,thetasN*Ngauss_ang,-1])
  #          tf.debugging.check_numerics(exp_ang_theta_ijk, message='combine cos and angular gaussian contains NaN')

            #cutoff fc(rij)*fc(rik) from nat x Nneigh to nat x Nneigh x Nneigh and reshape to nat x Nneigh**2
            
            #nat x Nneigh x Nneigh
            fc_rij_rik = self.tf_fcut(all_rij_norm, rc_ang)[:,tf.newaxis,:] * self.tf_fcut(all_rij_norm, rc_ang)[:,:,tf.newaxis]
            #fc_rij_rik = tf.gather_nd(fc_rij_rik_all, cond)
            #fc_rij_rik = tf.reshape(tf.RaggedTensor.from_value_rowids(fc_rij_rik, cond[:,0]).to_tensor(), [nat,-1])
   #         tf.debugging.check_numerics(fc_rij_rik, message='cutoff contains NaN')
            
            #convert species_encoder from Nneig=nat*n_replicas to nat,Nneig x Nneig to have i,j,k
            #Nneigh x Nneigh
            species_encoder_jk = species_encoder[:,tf.newaxis,:] * species_encoder[:,:,tf.newaxis]
            #species_encoder_jk = tf.tile()
           # species_encoder_jk = tf.gather_nd(species_encoder_jk, cond)
           # species_encoder_jk = tf.reshape(tf.RaggedTensor.from_value_rowids(species_encoder_jk, cond[:,0]).to_tensor(), [nat,-1])

            #collect all terms
            spec_encoder_fcjk = species_encoder_jk * fc_rij_rik
            #nat x Neigh**2
            spec_encoder_fcjk = tf.reshape(spec_encoder_fcjk, [nat, -1])
    #        tf.debugging.check_numerics(spec_encoder_fcjk, message='cutoff and spec contains NaN')
            _descriptor_ang = spec_encoder_fcjk[:,tf.newaxis,:] * exp_ang_theta_ijk
     #       tf.debugging.check_numerics(_descriptor_ang, message='Before the sum angular descriptors contains NaN')
            descriptor_ang = tf.reduce_sum(_descriptor_ang, axis=-1)
            tf.debugging.check_numerics(descriptor_ang, message='After the sum angular descriptors contains NaN')


            #descriptor_ang = descriptor_ang   
            #descriptor_ang = tf.where(tf.logical_or(tf.math.is_nan(descriptor_ang), tf.math.is_inf(descriptor_ang)),
            #                                      tf.zeros_like(descriptor_ang), descriptor_ang)
           #concatenate radial and angular descriptors
            #tf.debugging.check_numerics(descriptor_ang, message='angular descriptors contains NaN')
            #descriptor_ang = tf.ones((nat,thetasN*Ngauss_ang))
            atomic_descriptors = tf.concat([atomic_descriptors, descriptor_ang], axis=1)
            
            #feature_size = Ngauss + Ngauss_ang * thetasN
            #the descriptors can be scaled
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])
            #mutiply by independent weights
            atomic_descriptors = species_encoder0[:,tf.newaxis] * atomic_descriptors
            #predict energy and forces

            atomic_energies = self.atomic_nets(atomic_descriptors)
            #atomic_energies = tf.where(atomic_energies < )
            #print(tf.shape(atomic_energies), tf.shape(atomic_descriptors))
            #atomic_energies = tf.reshape(atomic_energies, [-1])
            #total_energy = tf.reduce_sum(tf.where(tf.logical_or(tf.math.is_nan(atomic_energies), tf.math.is_inf(atomic_energies)),
            #                                      tf.zeros_like(atomic_energies), atomic_energies))
            total_energy = tf.reduce_sum(atomic_energies)
           # print(total_energy)
       # print(positions, positions)   
        forces = g.gradient(total_energy, positions)
        
        #tf.debugging.check_numerics(total_energy, message='Total_energy contains NaN')
        #tf.debugging.check_numerics(positions, message='positions contains NaN')
        #tf.debugging.check_numerics(forces, message='forces contains NaN')
        #print(forces)
        #forces = tf.reshape(forces, [nat,3]) 
        #forces = tf.cast(forces, tf.float32)
        padding = tf.zeros((nmax_diff,3))
        forces = tf.concat([forces, padding], 0)
        #forces = tf.zeros((nmax_diff+nat,3))   
        return [tf.cast(total_energy, tf.float32), -tf.cast(forces, tf.float32)]

    #def compute_loss(self,x, y, y_pred):
    #    loss = tf.reduce_mean((y_pred - y) ** 2)
    #    loss += tf.reduce_sum(self.losses)
    #    self.loss_tracker.update_state(loss)
    #    return loss
    
    #def reset_metrics(self):
    #    self.loss_tracker.reset_state()
    #def predict_total_energy(self, input_data, mask, atomic_nets):
        
    #    nmax = mask.shape[1]

        #nfeature = input_data.shape[-1]

    #    atomic_nets = Networks(self.feature_size, self.layer_sizes)
       # print(atomic_nets.shape)
        #print(nmax, nfeature)
        #predict all peratom energies
    #    atomic_energies = atomic_nets(tf.reshape(input_data, (-1,self.feature_size)))

        #atomic_energies = tf.reshape(atomic_energies, (-1, nmax))

        #mask out physical atoms
        #total_energy = tf.reduce_sum(atomic_energies*mask, axis=-1)
        #print(total_energy)
        #return total_energy
    #@property
    #def width(self):
    #    return self._width
    #@width.setter
    #def width(self, value):
    #    self._width = value

        
            
    def call(self, inputs, training=False):
        '''inpu has a shape of batch_size x nmax_atoms x feature_size'''
        # may be just call the energy prediction here which will be needed in the train and test steps
        
        # the input are going to be filename from which descriptors and targets are going to be extracted
        rcuts = tf.tile([self.rcut], [self.batch_size])
        rcuts_ang = tf.tile([self.rcut_ang], [self.batch_size])
        batch_RsN_rad = tf.tile([self.RsN_rad], [self.batch_size])
        batch_RsN_ang = tf.tile([self.RsN_ang], [self.batch_size])
       
        batch_thetaN = tf.tile([self.thetaN], [self.batch_size])
            
        if self._params_trainable:
            self.width_value = tf.reshape(self.width_nets(tf.constant([self._width])), [-1])
            self.width_value += self._width
            
            self.width_value_ang = tf.reshape(self.width_nets_ang(tf.constant([self.width_ang])), [-1])
            self.width_value_ang += self.width_ang
            self.zeta_value = tf.reshape(self.zeta_nets(tf.constant([self.zeta])), [-1])
            self.zeta_value += self.zeta
        else:
            self.width_value = tf.constant([self._width], dtype=tf.float32)
            self.width_value_ang = tf.constant([self.width_ang], dtype=tf.float32)
            self.zeta_value = tf.constant([self.zeta], dtype=tf.float32)
        
        batch_width = tf.tile(self.width_value, [self.batch_size])
        batch_width_ang = tf.tile(self.width_value_ang, [self.batch_size])
        batch_zeta = tf.tile(self.zeta_value, [self.batch_size])
        
        
        
        #positions = tf.reshape(inputs[0], (self.batch_size, -1))
        #species_encoder = tf.reshape(inputs[1], (self.batch_size, -1))
        
        batch_nats = inputs[2]
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        batch_nmax = tf.tile([nmax], [self.batch_size])
        #print(batch_nats, batch_nmax)
        nmax_diff = batch_nmax - batch_nats
        
        #positions and species_encoder are ragged tensors are converted to tensors before using them
        positions = tf.reshape(inputs[0].to_tensor(shape=(-1,nmax,3)), (-1, 3*nmax))
        #obtain species encoder
        spec_identity = tf.constant(self.species_identity, dtype=tf.int32) - 1

        species_one_hot_encoder = tf.one_hot(spec_identity, depth=self.nelement)
            
        trainable_species_encoder = self.species_nets(species_one_hot_encoder)
        species_encoder = inputs[1].to_tensor(shape=(-1, nmax))
        batch_species_encoder = tf.zeros([self.batch_size, nmax], dtype=tf.float32)

        for idx, spec in enumerate(self.species_identity):
            values = tf.ones([self.batch_size, nmax], dtype=tf.float32) * trainable_species_encoder[idx]
            batch_species_encoder += tf.where(tf.equal(species_encoder,tf.cast(spec,tf.float32)), 
                    values, tf.zeros([self.batch_size, nmax]))

        cells = inputs[3]
        replica_idx = inputs[4]
        
        elements = (rcuts, batch_RsN_rad, batch_species_encoder, batch_width,
                positions, nmax_diff, batch_nats, 
                batch_zeta, rcuts_ang, batch_RsN_ang, 
                batch_thetaN, batch_width_ang, cells, replica_idx)

        energies, forces = tf.map_fn(self.tf_predict_energy_forces, elements, fn_output_signature=[tf.float32, tf.float32])
        
        #energies = energies_forces[:,0]
        #forces = tf.reshape(energies_forces[:,1:], [-1, nmax, 3])
        #forces = tf.zeros((self.batch_size,nmax,3))
        return energies, forces, tf.reduce_sum(self.width_value),tf.reduce_sum(self.width_value_ang), tf.reduce_sum(self.zeta_value)
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs_target = data
        inputs = inputs_target[:5]
        target = inputs_target[5]
        target_f = tf.reshape(inputs_target[6].to_tensor(), [-1])
        target_f = tf.cast(target_f, tf.float32)
        
        #in_shape = inputs[0].shape
        
        # the last dimension is the feature size
        #inputs = tf.reshape(inputs, [-1, in_shape[-1]])
        
        batch_nats = tf.cast(inputs[2], tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        
        with tf.GradientTape() as tape:
            e_pred, forces, width, width_ang, zeta = self(inputs, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            ediff = (e_pred - target)
            forces = tf.reshape(forces, [-1])

            emse_loss = tf.reduce_mean((ediff/batch_nats)**2)

            dforces = tf.reshape(target_f, [self.batch_size, 3*nmax]) - tf.reshape(forces, [self.batch_size, 3*nmax])
            fmse_loss = tf.reduce_mean(tf.reduce_sum((dforces)**2, axis=-1) / (3*batch_nats))

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

        mae_f = tf.reduce_mean(tf.reduce_sum(tf.abs(dforces), axis=-1) / (3*batch_nats))
        rmse_f = tf.sqrt(tf.reduce_mean(tf.reduce_sum((dforces)**2, axis=-1) / (3*batch_nats)))
        
        
        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})
        
        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})
        metrics.update({'Parameter-width-rad':width})
        metrics.update({'Parameter-width-ang':width_ang})
        metrics.update({'Parameter-zeta':zeta})
        
        #    if self._f_loss:
        #        metrics.update({'F_MAE': Fmae})
        #if 'RMSE' in self._printmetrics:
        #    metrics.update({'RMSE/at': rmseat})
        #    if self._f_loss:
        #        metrics.update({'F_RMSE': Frmse})
        
        metrics.update({'loss': loss})
        metrics.update({'energy loss': emse_loss})
        metrics.update({'force loss': fmse_loss})
        
        #with writer.set_as_default():
           
        tf.summary.scalar('1. Losses/1. Total',loss,self._train_counter)
        tf.summary.scalar('2. Metrics/1. RMSE/atom',rmse,self._train_counter)
        tf.summary.scalar('2. Metrics/2. MAE/atom',mae,self._train_counter)
        tf.summary.scalar('2. Metrics/3. RMSE_F',rmse_f,self._train_counter)
        tf.summary.scalar('2. Metrics/3. MAE_F',mae_f,self._train_counter)
        tf.summary.scalar('3. Parameters/1. width',width,self._train_counter)
        tf.summary.scalar('3. Parameters/2. width_ang',width_ang,self._train_counter)
        tf.summary.scalar('3. Parameters/3. zeta',zeta,self._train_counter)
            


          

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
        inputs = inputs_target[:5]
        target = inputs_target[5]
        target_f = tf.reshape(inputs_target[6].to_tensor(), [-1])
        target_f = tf.cast(target_f, tf.float32)
        
        #in_shape = inputs[0].shape
        e_pred, forces,width, width_ang, zeta = self(inputs, training=True)  # Forward pass
        
        forces = tf.reshape(forces, [-1])
        # Update metrics (includes the metric that tracks the loss)
        
        batch_nats = tf.cast(inputs[2], tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        
        ediff = (e_pred - target)
        
        mae = tf.reduce_mean(tf.abs(ediff / batch_nats))
        rmse = tf.sqrt(tf.reduce_mean((ediff/batch_nats)**2))
        
        #mae_f = tf.reduce_mean(tf.abs(target_f - forces))
        #rmse_f = tf.sqrt(tf.reduce_mean((target_f - forces)**2))
        dforces = tf.reshape(target_f, [self.batch_size, 3*nmax]) - tf.reshape(forces, [self.batch_size, 3*nmax]) 
        mae_f = tf.reduce_mean(tf.reduce_sum(tf.abs(dforces), axis=-1) / (3*batch_nats))
        rmse_f = tf.sqrt(tf.reduce_mean(tf.reduce_sum((dforces)**2, axis=-1) / (3*batch_nats)))
        

        metrics = {}
        #mae = tf.reduce_mean(tf.abs(ediff))
        loss = rmse * rmse + rmse_f * rmse_f
        #rmse = tf.sqrt(loss)
        
        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})
        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})
        
        #    if self._f_loss:
        #        metrics.update({'F_MAE': Fmae})
        #if 'RMSE' in self._printmetrics:
        #    metrics.update({'RMSE/at': rmseat})
        #    if self._f_loss:
        #        metrics.update({'F_RMSE': Frmse})
        
        metrics.update({'loss': loss})
          
        return {key: metrics[key] for key in metrics.keys()}


    def predict_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs_target = data
        inputs = inputs_target[:5]
        target = inputs_target[5]
        target_f = tf.reshape(inputs_target[6].to_tensor(), [-1])
        target_f = tf.cast(target_f, tf.float32)
        #in_shape = inputs[0].shape
        e_pred, forces, width, width_ang, zeta = self(inputs, training=False)  # Forward pass
        
        _forces = tf.identity(forces)
        _forces = tf.reshape(_forces, [-1])
        
        batch_nats = tf.cast(inputs[2], tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        ediff = (e_pred - target)
        
    
        mae = tf.reduce_mean(tf.abs(ediff / batch_nats))
        rmse = tf.sqrt(tf.reduce_mean((ediff/batch_nats)**2))
        
        dforces = tf.reshape(target_f, [self.batch_size, 3*nmax]) - tf.reshape(forces, [self.batch_size, 3*nmax]) 
        mae_f = tf.reduce_mean(tf.reduce_sum(tf.abs(dforces), axis=-1) / (3*batch_nats))
        rmse_f = tf.sqrt(tf.reduce_mean(tf.reduce_sum((dforces)**2, axis=-1) / (3*batch_nats)))

        metrics = {}
        #mae = tf.reduce_mean(tf.abs(ediff))
        #loss = tf.reduce_mean(ediff**2)
        #rmse = tf.sqrt(loss)
        
        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})
        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})
        
        return [target, e_pred, metrics, target_f, forces, batch_nats]


# Construct and compile an instance of CustomModel

def convert_json2ASE_atoms(atomic_energy, file):
    Ry2eV = 13.6057039763
    from ase import Atoms
    import json
    data = json.load(open(file))
    try:
        idx, symbols, positions, forces = zip(*data['atoms'])
    except:
        idx, symbols, positions, forces, charge = zip(*data['atoms'])

    try:
        cell = np.asarray(data['lattice_vectors'])
        pbc = True
    except:
        cell = None
        pbc=False
    E0 = 0.0
    if len(atomic_energy)>0:
        for i,sp in enumerate(species):
            Nsp = len(np.where(np.asarray(symbols).astype(str)==sp)[0])
            E0 += Nsp * atomic_energy[i]
   # print(E0)

    positions = np.asarray(positions).astype(float)
    forces = np.asarray(forces).astype(float)
#    encoder = all_species_encoder
    _spec_encoder = np.asarray([species_encoder(ss) for ss in symbols])

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
    atoms.info = {'energy':energy-E0}

    return atoms
def atomic_number(symbol):

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
                'Lr','Rf','Db','Sg','Bh','Hs','Mt' ]

    return symbols.index(symbol)+1

#def species_encoder(species):
#    atomic_numbers = [element(s).atomic_number for s in species]
#    #standerdize the atomic numbers
#    if len(species) == 1:
#        return {species[0]:1.0}
#    meanz = np.mean(atomic_numbers)
#    stdz = np.std(atomic_numbers)

#    atomic_numbers = np.asarray(atomic_numbers).astype(float)
#    atomic_numbers = (atomic_numbers-meanz)/stdz
    #atomic_numbers = np.abs(atomic_numbers)
#    return {s:atomic_numbers[species.index(s)] for s in species}
'''def _species_encoder(species):
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

    atomic_numbers = [symbols.index(s) for s in symbols]
    
    #standerdize the atomic numbers
 
    meanz = np.mean(atomic_numbers)
    stdz = np.std(atomic_numbers)

    atomic_numbers = np.asarray(atomic_numbers).astype(float)
    atomic_numbers = (atomic_numbers - meanz) / stdz
    atomic_numbers_std = {s:atomic_numbers[symbols.index(s)] for s in symbols}
    #atomic_numbers = np.abs(atomic_numbers)
    #return atomic_numbers_std, {s:atomic_numbers_std[s] for s in species}, len(symbols)
    return {s:atomic_numbers_std[s] for s in species}
'''
def species_encoder(species):
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

    #atomic_numbers = [symbols.index(s) for s in symbols]
    
    #standerdize the atomic numbers
 
    #meanz = np.mean(atomic_numbers)
    #stdz = np.std(atomic_numbers)

    #atomic_numbers = np.asarray(atomic_numbers).astype(float)
    #atomic_numbers = (atomic_numbers - meanz) / stdz
    #atomic_numbers_std = {s:atomic_numbers[symbols.index(s)] for s in symbols}
    #atomic_numbers = np.abs(atomic_numbers)
    return symbols.index(species)
    #return {s:atomic_numbers_std[s] for s in species}

###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import numpy as np
import itertools

def orthogonal_vector(input_vector):
    """ calculate a random orthogonal unit vector to input_vector
        by solving the equation ax * x + ay * y + az * z = 0
        with input_vector = [ax, ay, az]
        candidate_vector = [x, y, z]

    Parameters
    ----------
    input_vector : 1D array
        a vector

    Returns
    -------
    1D array
        candidate_vector, random unit vector
    """
    candidate_vector = np.zeros(3)
    count = 0  # check for first non-zero element of input_vector
    for idx in range(3):

        if input_vector[idx] != 0 and count < 1:
            # choose only one of the components of candidate vector and the
            # other one is zero (this is sufficient !)
            candidate_vector[(idx + 1) % 3] = np.random.random(1)
            # candidate_vector[(idx + 2) % 3] = np.random.random(1)

            candidate_vector[idx] = -(candidate_vector[(idx + 1) % 3]
                                      * input_vector[(idx + 1) % 3]
                                      + candidate_vector[(idx + 2) % 3]
                                      * input_vector[(idx + 2) % 3]) \
                / input_vector[idx]
            count += 1

    candidate_vector = candidate_vector / np.linalg.norm(candidate_vector)

    return candidate_vector


def replicas_max_idx(lattice_vectors, Rc, pbc=[True, True, True]):
    """ calculate maximum number of cells needed with a given radial cutoff

    Parameters
    ----------
    lattice_vectors: lattice vectors as matrix (a1, a2, a3)
    Rc: maximum radial cutoff that you want to take in to account
    pbc: Normally pbc are recovered from lattice vector,
         if in the lattice_vectors a direction is set to zero
         then no pbc is applied in that direction.
         This argument allow you to turn off specific directions
         in the case where that specific direction has a lattice vector
         greater then zero.
         To achieve this pass an array of 3 logical value (one for each
         direction). False value turn off that specific direction.
         Default is true in every direction.
         eg. pbc = [True, False, True] => pbc along a1 and a3

    Returns
    -------
    max_indices: [lmax, mmax, nmax], numpy array
       integers for the number of replicas.
    """
    if not isinstance(lattice_vectors, np.ndarray):
        _lattice_vectors = np.asarray(lattice_vectors)
    else:
        _lattice_vectors = lattice_vectors.copy()

    lattice_vector_lengths = np.linalg.norm(_lattice_vectors, axis=1)
    lattice_vector_lengths_bool = lattice_vector_lengths > 1e-6
    max_indices = np.zeros(3, dtype=int)
    if not lattice_vector_lengths_bool.any():
        return max_indices

    lattice_vectors_idxs = np.where(lattice_vector_lengths_bool)[0]
    lattice_vectors_idxs_false = np.where(~lattice_vector_lengths_bool)[0]

    # define an index control
    lat_vec_idx_control = len(lattice_vectors_idxs)

    for idx in lattice_vectors_idxs_false:

        # when only one vector is has a nonzero length
        if lat_vec_idx_control == 1:

            j_idx = lattice_vectors_idxs[0]
            # orthogonal vector to _lattice_vectors[j_idx
            _lattice_vectors[idx] = orthogonal_vector(_lattice_vectors[j_idx])

        # when two vectors have nonzero length/given
        # compute the  cross_product  of the two given vectors

        if lat_vec_idx_control == 2:
            _lattice_vectors[idx] = np.cross(
                _lattice_vectors[(idx - lat_vec_idx_control) % 3],
                _lattice_vectors[(idx - lat_vec_idx_control + 1) % 3])
            _lattice_vectors[idx] = _lattice_vectors[idx] \
                / np.linalg.norm(_lattice_vectors[idx])
        lat_vec_idx_control += 1

    reciprocal_vectors = np.linalg.inv(_lattice_vectors)
    reciprocal_vectors_length = np.linalg.norm(reciprocal_vectors, axis=0)

    for idx in lattice_vectors_idxs:
        if pbc[idx]:
            b_length = reciprocal_vectors_length[idx]
            max_indices[idx] = int(np.ceil(Rc * b_length))

    return max_indices


print('here')
def input_function(x, shuffle=True, batch_size=32): # inner function that will be returned
    dataset = tf.data.Dataset.from_tensor_slices(x)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.ragged_batch(batch_size,drop_remainder=True) # split dataset into batch_size batches and repeat process for num_epochs
    return dataset

def input_function_examples(data_path, shuffle=True, batch_size=32): # inner function that will be returned

    dataset = tf.data.Dataset.list_files(data_path+'*example')
    #dataset=dataset.shuffle(1000).batch(8).repeat(32)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size).repeat(num_epochs) # split dataset into batch_size batches and repeat process for num_epochs
    return dataset

def data_preparation(data_dir, species, data_format, 
                     energy_key, force_key,
                     rc_rad, rc_ang, pbc, batch_size, 
                     test_fraction=0.1,
                     atomic_energy=[]):
    
    rc = np.max([rc_rad, rc_ang])

    if data_format == 'panna_json':
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.split('.')[-1]=='example']
    elif data_format == 'xyz':
        # note, this is already atom object
        files = read(data_dir, index=':')
        #collect configurations
    #all_configs_ase = []
    all_positions = []
    all_species_encoder = []
    all_energies = []
    all_forces = []
    all_natoms = []
    cells = []
    replica_idx = []
    #  atoms = convert_json2ASE_atoms(files[0],species)
    #  e0 = atoms.info['energy'] / len(atoms.positions)
    #  print(e0)
    #implement multiprocessing
    #species encoder for all atomic species.
    #_spec_encoder = species_encoder()
    
    species_identity = [species_encoder(s) for s in species]

    #partial_convert = convert_json2ASE_atoms(all_species_encoder,atomic_energy)
    #number of precesses
    #p = Pool(num_process)
    #Ndata = len(files)

    
    for file in files:
        if data_format == 'panna_json':
            atoms = convert_json2ASE_atoms(atomic_energy,file)
        elif data_format == 'xyz':
            atoms = file
            symbols = list(atoms.symbols)
            #encoder = species_encoder(species)

            _encoder = np.asarray([_spec_encoder[ss] for ss in symbols])
            atoms.new_array('encoder', _encoder)

            
    #    if atoms.info['energy'] > 30.0:
    #        continue
        #all_configs_ase.append(atoms)
        all_energies.append(atoms.info['energy'])
        all_positions.append(atoms.positions)
        all_species_encoder.append(atoms.get_array('encoder'))
        try:
            all_forces.append(atoms.get_array('forces'))
        except:
            all_forces.append(atoms.get_forces())

        all_natoms.append(atoms.get_global_number_of_atoms())
        cells.append(atoms.cell)
        replica_idx.append(replicas_max_idx(atoms.cell, rc, pbc=pbc))

    Ntest = int(test_fraction*len(all_natoms))

    cells_test = tf.constant(cells[:Ntest])
    cells_train = tf.constant(cells[Ntest:])
    replica_idx_test = tf.constant(replica_idx[:Ntest])
    replica_idx_train = tf.constant(replica_idx[Ntest:])

    all_positions_test = tf.ragged.constant(all_positions[:Ntest])
    all_positions_train = tf.ragged.constant(all_positions[Ntest:])

    #forces
    all_forces_test = tf.ragged.constant(all_forces[:Ntest])
    all_forces_train = tf.ragged.constant(all_forces[Ntest:])

    #print(Ntest, len(all_positions_train), len(all_positions_test))
    all_species_encoder_test = tf.ragged.constant(all_species_encoder[:Ntest], dtype=tf.float32)
    all_species_encoder_train = tf.ragged.constant(all_species_encoder[Ntest:], dtype=tf.float32)

    all_natoms_test = tf.constant(all_natoms[:Ntest])
    all_natoms_train = tf.constant(all_natoms[Ntest:])

    all_energies_test = tf.constant(all_energies[:Ntest])
    all_energies_train = tf.constant(all_energies[Ntest:])



    train_data = input_function((all_positions_train, all_species_encoder_train,
                                 all_natoms_train,cells_train, replica_idx_train,
                                 all_energies_train, all_forces_train),
                                shuffle=True, batch_size=batch_size)

    test_data = input_function((all_positions_test, all_species_encoder_test,
                                all_natoms_test, cells_test,replica_idx_test,
                                all_energies_test, all_forces_test),
                                shuffle=True, batch_size=batch_size)


    return [train_data, test_data, species_identity]


def create_model(config_file):


    #Read in model parameters
    #I am parsing yaml files with all the parameters
    #
    layer_sizes = configs['layer_sizes']
    save_freq = configs['save_freq']
    zeta = configs['zeta']
    thetaN = configs['thetaN']
    RsN_rad = configs['RsN_rad']
    RsN_ang = configs['RsN_ang']
    rc_rad = configs['rc_rad'] 
    rc_ang = configs['rc_ang'] 
    #estimate initial parameters
    width_ang = RsN_ang * RsN_ang / (rc_ang-0.25)**2
    width = RsN_rad * RsN_rad / (rc_rad-0.25)**2
    fcost = configs['fcost']
    #trainable linear model
    params_trainable = configs['params_trainable']
    pbc = configs['pbc'] 
    if pbc:
        pbc = [True,True,True]
    else:
        pbc = [False,False,False]
    initial_lr = configs['initial_lr']
    #this is the global step
    decay_step = configs['decay_step']
    decay_rate = configs['decay_rate']
    #activations are basically sigmoid and linear for now
    activations = ['sigmoid', 'sigmoid', 'linear']
    species = configs['species']
    batch_size = configs['batch_size']
    model_outdir = configs['outdir']
    num_epochs = configs['num_epochs']
    data_dir = configs['data_dir']
    data_format = configs['data_format']
    energy_key = configs['energy_key']
    force_key = configs['force_key']
    test_fraction = configs['test_fraction']
    try:
        atomic_energy = configs['atomic_energy']
    except:
        atomic_energy = []
    
    train_data, test_data, species_identity = data_preparation(data_dir, species, data_format,
                     energy_key, force_key,
                     rc_rad, rc_ang, pbc, batch_size,
                     test_fraction=test_fraction,
                     atomic_energy=atomic_energy)

    model = mBP_model(layer_sizes,
                      rc_rad, species_identity, width, batch_size,
                       params_trainable, activations,
                      rc_ang,RsN_rad,RsN_ang,
                      thetaN,width_ang,zeta,
                      fcost=fcost,
                      pbc=pbc)

    initial_learning_rate = initial_lr

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    

    # Create a callback that saves the model's weights every 5 epochs
    if not os.path.exists(model_outdir):
        os.mkdir(model_outdir)
    checkpoint_path = model_outdir+"/models/ckpts-{epoch:04d}.ckpt"

    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                    save_freq=save_freq,
                                                    options=None),
                   tf.keras.callbacks.TensorBoard(model_outdir, histogram_freq=1),
                   tf.keras.callbacks.BackupAndRestore(backup_dir=model_outdir+"/tmp_backup"),
                   tf.keras.callbacks.CSVLogger(model_outdir+"/metrics.dat", separator=" ", append=True)]


    model.save_weights(checkpoint_path.format(epoch=0))
    #train the model
    
        
    #load the last saved epoch

    model.compile(optimizer=optimizer, loss="mse", metrics=["MAE", 'loss'])
    try:
        model.fit(train_data,
             epochs=num_epochs,
             batch_size=batch_size,
             validation_data=test_data,
             validation_freq=10,
             callbacks=[cp_callback])
    except:
      pass

    model.fit(train_data,
              epochs=num_epochs,
              batch_size=batch_size,
             validation_data=test_data,
             validation_freq=10,
             callbacks=[cp_callback])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='create ML model')
    parser.add_argument('-c', '--config', type=str,
                        help='configuration file', required=True)
    args = parser.parse_args()


    import yaml
    with open(args.config) as f:
        configs = yaml.safe_load(f)

    print(configs)

#    configs = {}
#    num_epochs = configs['num_epochs']
#    batch_size = configs['batch_size']
#    data_dir = configs['data_dir']
#    data_format = configs['data_format']
#    species = configs['species']
#    energy_key = configs['energy_key']
#    force_key = configs['force_key']
#    test_fraction = configs['force_key']
#    layer_sizes = configs['layer_sizes']
#    save_freq = configs['save_freq']
#    zeta = configs['zeta']
#    thetaN = configs['thetaN']
#    RsN_rad = configs['RsN_rad']
#    RsN_ang = configs['RsN_ang']
#    rc_rad = configs['rc_rad']
#    rc_ang = configs['rc_ang']
#    fcost = configs['fcost']
    #trainable linear model
#    params_trainable = configs['params_trainable']
#    pbc = configs['pbc']
#    initial_lr = configs['lr']
#    #this is the global step
#    decay_step = configs['decay_step']
#    decay_rate = configs['decay_rate']
#    model_outdir = configs['outdir']


    create_model(configs)
    



