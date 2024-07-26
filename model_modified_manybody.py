from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from IPython.display import clear_output
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
import mendeleev
from mendeleev import element
import math 
import itertools, os
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

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
                                        kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        bias_regularizer=regularizers.L1L2(l1=l1,l2=l2),
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
                                        kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        bias_regularizer=regularizers.L1L2(l1=l1,l2=l2),
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
                ecost=1.0,
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
                rmax_d=12.0,
                body_order=3,
                Nzeta=None,
                learnable_centers=True,
                variable_width=False):
        
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
        self.body_order = body_order
        
        base_size = self.RsN_ang * self.thetaN
        self.feature_size = self.RsN_rad + base_size
        if self.body_order >= 4:
            for k in range(3,self.body_order):
                self.feature_size += base_size

        self.pbc = pbc
        self.fcost = float(fcost)
        self.ecost = float(ecost)
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
        self.learnable_centers = learnable_centers
        self.variable_width = variable_width
     
        self.atomic_nets = Networks(self.feature_size, self.layer_sizes, self._activations, l1=self.l1, l2=self.l2)

        # the number of elements in the periodic table
        self.nelement = nelement
        # create a species embedding network with 1 hidden layer Nembedding x Nspecies
        self.species_nets = Networks(self.nelement, [self.nspec_embedding,1], ['sigmoid','sigmoid'], prefix='species_encoder')

        constraint = None
        if self._params_trainable:
            Nwidth_rad = 1
            Nwidth_ang = 1
            if self.variable_width:
                Nwidth_rad = self.RsN_rad
                Nwidth_ang = self.RsN_ang

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
            if self.train_zeta:
                init = tf.keras.initializers.RandomNormal(mean=self.zeta, stddev=0.05)
                self.zeta_nets = Networks(1, [1], ['softplus'],
                                      weight_initializer=init,
                                      bias_initializer='zeros',
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='zeta')
        constraint = None
        if self.learnable_centers:
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
            init = tf.keras.initializers.GlorotNormal(seed=56789)
            self.thetas_nets = Networks(1, [self.thetaN], ['sigmoid'],
                                          weight_initializer=init,
                                          bias_initializer=init,
                                          kernel_constraint=constraint,
                                          bias_constraint=constraint, prefix='thetas')

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

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.float32)])
    def tf_fcut(self,r,rc):
        dim = tf.shape(r)
        pi = tf.constant(math.pi, dtype=tf.float32)
        return tf.where(tf.logical_and(r<=rc,r>1e-8), 0.5*(1.0 + tf.cos(pi*r/rc)), tf.zeros(dim, dtype=tf.float32))
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.float32)])
    def _tf_fcut(self,r,rc):
        dim = tf.shape(r)
        #pi = tf.constant(math.pi, dtype=tf.float32)
        x = tf.tanh(1 - r / rc)
        return tf.where(tf.logical_and(r<=rc,r>1e-8), x*x*x, tf.zeros(dim, dtype=tf.float32))
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
                input_signature=[(
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(3,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32)
                )])
    def tf_predict_energy_forces(self,x):

        rc = tf.cast(self.rcut,dtype=tf.float32)
        Ngauss = tf.cast(self.RsN_rad, dtype=tf.int32)
        rc_ang = tf.cast(self.rcut_ang, dtype=tf.float32)
        Ngauss_ang = tf.cast(self.RsN_ang, dtype=tf.int32)
        thetasN = tf.cast(self.thetaN, dtype=tf.int32)



        nat = tf.cast(x[4], dtype=tf.int32)
        nmax_diff = tf.cast(x[3], dtype=tf.int32)
        species_encoder = tf.cast(x[0][:nat], tf.float32)
        width = tf.cast(x[1], dtype=tf.float32)
        positions = tf.reshape(x[2][:nat*3], [nat,3])
        positions = tf.cast(positions, tf.float32)

        #zeta cannot be a fraction
        zeta = tf.cast(x[5], dtype=tf.float32)
        width_ang = tf.cast(x[6], dtype=tf.float32)
        
        cell = tf.cast(x[7], tf.float32)
        replica_idx = tf.cast(x[8], tf.float32)
        Rs = tf.cast(x[9], tf.float32)
        Rs_ang = tf.cast(x[10], tf.float32)
        theta_s = tf.cast(x[11], tf.float32)
        evdw = 0.0
        
        if self.include_vdw:
            C6 = tf.cast(x[12], tf.float32)

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
                if self.include_vdw:
                    C6_extended = tf.reshape(tf.tile([C6], [1, n_replicas]), [nat*n_replicas])
            else:
                positions_extended = tf.identity(positions)
                species_encoder0 = tf.identity(species_encoder)
                if self.include_vdw:
                    C6_extended = tf.identity(C6)
            #positions_extended has nat x nreplica x 3 for periodic systems and nat x 1 x 3 for molecules
            
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
            if self.variable_width:
                gauss_args = width[tf.newaxis,:,tf.newaxis] * (all_rij_norm[:,tf.newaxis,:] - Rs[tf.newaxis,:,tf.newaxis])**2
            else:
                gauss_args = width * (all_rij_norm[:,tf.newaxis,:] - Rs[tf.newaxis,:,tf.newaxis])**2

            #this is no needed because have no contribution from atoms outside cutoff because of fc
            #mask the non-zero values after ragged tensor is converted to fixed shape
            # This is very important to avoid adding contributions from atoms outside the sphere
            #nonzero_val = tf.where(all_rij_norm > 1e-8, 1.0,0.0)
            #nonzero_val = tf.cast(nonzero_val, tf.float32)
            #gauss_args = gauss_args * nonzero_val[:,tf.newaxis,:]



            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x 1 x Nneigh
            #fcuts is nat x Nneigh and reshaped to nat x 1 x Nneigh
            args = species_encoder_rad[:,tf.newaxis,:] * self.tf_app_gaussian(gauss_args) * self.tf_fcut(all_rij_norm, rc)[:,tf.newaxis,:]
            
            # sum over neighbors j including periodic boundary conditions
            atomic_descriptors = tf.reduce_sum(args, axis=-1)



            #implement angular part: compute vect_rij dot vect_rik / rij / rik
                   
            inball_ang = tf.where(all_rij_norm <=rc_ang)
            
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
            #nat x Nneigh X Nneigh
            #cos_theta_ijk = tf.matmul(rij_unit, tf.transpose(rij_unit, (0,2,1)))
            #rij.rik
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
            cos_theta_ijk = tf.reshape(cos_theta_ijk, [nat,Nneigh,Nneigh])
            
            #clip values to avoid divergence in the derivative with the division by sin(theta)
            reg2 = 1e-6

            cos_theta_ijk = tf.clip_by_value(cos_theta_ijk, clip_value_min=-1.0+reg2, clip_value_max=1.0-reg2)

            cos_theta_ijk2 = cos_theta_ijk * cos_theta_ijk
            #nat x thetasN x N_unique
            sin_theta_ijk = tf.sqrt(1.0 - cos_theta_ijk2)
            _cos_theta_ijk_theta_s = cos_theta_ijk[:,:,:, tf.newaxis] * cos_theta_s[tf.newaxis,tf.newaxis,tf.newaxis,:] 
            _sin_theta_ijk_theta_s = sin_theta_ijk[:,:,:, tf.newaxis] * sin_theta_s[tf.newaxis,tf.newaxis,tf.newaxis,:]
            cos_theta_ijk_theta_s = 1.0 + _cos_theta_ijk_theta_s + _sin_theta_ijk_theta_s
            #tf.debugging.check_numerics(cos_theta_ijk_theta_s, message='cos_theta_ijk_theta_s contains NaN')
            #cos_theta_ijk_theta_s += tensors_contrib
            
            norm_ang = 2.0 
            #Nat,Nneigh, Nneigh,ThetasN
            cos_theta_ijk_theta_s_zeta = (cos_theta_ijk_theta_s / norm_ang)**zeta
            #tf.debugging.check_numerics(cos_theta_ijk_theta_s_zeta, message='cos_theta_ijk_theta_s_zeta contains NaN')

            if self.variable_width:
                rik_norm_rs = width_ang[tf.newaxis,tf.newaxis,:] * (all_rij_norm[:,:,tf.newaxis]  - Rs_ang[tf.newaxis,tf.newaxis,:])**2
            else:
                rik_norm_rs = width_ang * (all_rij_norm[:,:,tf.newaxis]  - Rs_ang[tf.newaxis,tf.newaxis,:])**2

            #compute the exponetial term: Nat, Nneigh, RNs_ang
            radial_ij = self.tf_app_gaussian(rik_norm_rs) * self.tf_fcut(all_rij_norm, rc_ang)[:,:,tf.newaxis] * \
                           species_encoder[:,:,tf.newaxis]
            
            # dimension = [Nat, Nneigh, Nneigh, Ngauss_ang, thetaN]
            exp_ang_theta_ijk = cos_theta_ijk_theta_s_zeta[:,:,:,tf.newaxis,:] * radial_ij[:,:,tf.newaxis,:,tf.newaxis]
            exp_ang_theta_ijk = tf.reshape(exp_ang_theta_ijk, [nat,Nneigh,Nneigh,Ngauss_ang,thetasN])
            
            #dimension = [Nat, Nneigh, Ngauss_ang, thetaN]
            Base_vector_ij_s = tf.reduce_sum(exp_ang_theta_ijk, axis=2)
            Base_vector_ij_s = tf.reshape(Base_vector_ij_s, [nat, Nneigh, -1])
            
            radial_ij = tf.reshape(tf.tile(radial_ij[:,:,:,tf.newaxis],[1,1,1,thetasN]), [nat,Nneigh,-1])
            body_descriptor_3 = tf.reduce_sum(Base_vector_ij_s * radial_ij, axis=1) #sum over neigh and Rs_ang
            body_descriptor_3 = tf.reshape(body_descriptor_3, [nat, Ngauss_ang*thetasN])
            
            atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_3], axis=1)
            if self.body_order >= 4:
                body_tensor_4 = Base_vector_ij_s * Base_vector_ij_s
                body_descriptor_4 = tf.reduce_sum(body_tensor_4 * radial_ij, axis=1)
                body_descriptor_4 = tf.reshape(body_descriptor_4, [nat, Ngauss_ang*thetasN])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_4], axis=1)
            if self.body_order >= 5:
                body_tensor_5 = body_tensor_4 * Base_vector_ij_s
                body_descriptor_5 = tf.reduce_sum(body_tensor_5 * radial_ij, axis=1)
                body_descriptor_5 = tf.reshape(body_descriptor_5, [nat, Ngauss_ang*thetasN])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_5], axis=1)



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

        forces = g.gradient(total_energy, positions)
        
        padding = tf.zeros((nmax_diff,3))
        forces = tf.concat([forces, padding], 0)
        #forces = tf.zeros((nmax_diff+nat,3))   
        return [tf.cast(total_energy, tf.float32), -tf.cast(forces, tf.float32)]
    def call(self, inputs, training=False):
        '''inpu has a shape of batch_size x nmax_atoms x feature_size'''
        # may be just call the energy prediction here which will be needed in the train and test steps
        # the input are going to be filename from which descriptors and targets are going to be extracted

        batch_size = tf.shape(inputs[2])[0]

        if self._params_trainable:
            inputs_width = tf.ones(1)
            self.width_value = tf.reshape(self.width_nets(inputs_width[tf.newaxis, :]), [-1])

            inputs_width_ang = tf.ones(1)
            self.width_value_ang = tf.reshape(self.width_nets_ang(inputs_width_ang[tf.newaxis, :]), [-1])
            if self.train_zeta:
                inputs_zeta = tf.ones(1)
                self.zeta_value = tf.reshape(self.zeta_nets(inputs_zeta[tf.newaxis, :]), [-1])
            else:
                self.zeta_value = tf.ones(self.thetaN) * self.zeta
        else:
            self.width_value = tf.ones(self.RsN_rad) * self._width
            self.width_value_ang = tf.ones(self.RsN_ang) * self.width_ang
            self.zeta_value = tf.ones(self.thetaN) * self.zeta

                
        #inputs for center networks
        tf_pi = tf.constant(math.pi, dtype=tf.float32)
        if self.learnable_centers:
            Rs = tf.ones(1)
            Rs_ang = tf.ones(1)
            theta_s = tf.ones(1)
            Rs_rad_pred = tf.reshape(self.Rs_rad_nets(Rs[tf.newaxis,:]), [-1])
            rsmin = tf.reduce_min(Rs_rad_pred)
            rsmax = tf.reduce_max(Rs_rad_pred)
            #rescale between 0.5 and rc
            self._Rs_rad = 0.5 + (self.rcut - 0.5) * (Rs_rad_pred - rsmin) / (rsmax - rsmin + 1e-12)

            Rs_ang_pred = tf.reshape(self.Rs_ang_nets(Rs_ang[tf.newaxis,:]), [-1])
            rsmin = tf.reduce_min(Rs_ang_pred)
            rsmax = tf.reduce_max(Rs_ang_pred)
            #rescale between 0.5 and rc for the angular part
            self._Rs_ang = 0.5 + (self.rcut_ang - 0.5) * (Rs_ang_pred - rsmin) / (rsmax - rsmin + 1e-12)
            
            ts_pred = tf.reshape(self.thetas_nets(theta_s[tf.newaxis,:]), [-1])
            tsmin = tf.reduce_min(ts_pred)
            tsmax = tf.reduce_max(ts_pred)
            #rescale between 0 and pi
            self._thetas = (tf_pi - 0.0) * (ts_pred - tsmin) / (tsmax - tsmin + 1e-12)

            #self._Rs_ang = tf.reshape(self.Rs_ang_nets(Rs_ang[tf.newaxis,:]), [-1]) * self.rcut_ang
            #self._thetas = tf.reshape(self.thetas_nets(theta_s[tf.newaxis,:]), [-1]) * tf_pi
        else:
            self._Rs_rad = tf.range(self.RsN_rad, dtype=tf.float32) * (self.rcut)/tf.cast(self.RsN_rad, dtype=tf.float32)
            self._Rs_ang = tf.range(self.RsN_ang, dtype=tf.float32) * (self.rcut_ang)/tf.cast(self.RsN_ang, dtype=tf.float32)
            self._thetas = tf.range(self.thetaN, dtype=tf.float32) * tf_pi / tf.cast(self.thetaN, dtype=tf.float32)

        batch_width = tf.tile([self.width_value], [batch_size,1])
        batch_width_ang = tf.tile([self.width_value_ang], [batch_size,1])
        batch_zeta = tf.tile([self.zeta_value], [batch_size,1])
        batch_Rs_rad = tf.tile([self._Rs_rad], [batch_size,1])
        batch_Rs_ang = tf.tile([self._Rs_ang], [batch_size,1])
        batch_theta_s = tf.tile([self._thetas], [batch_size,1])


        batch_nats = inputs[2]
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        batch_nmax = tf.tile([nmax], [batch_size])
        nmax_diff = batch_nmax - batch_nats

        #positions and species_encoder are ragged tensors are converted to tensors before using them
        positions = tf.reshape(inputs[0].to_tensor(shape=(-1,nmax,3)), (-1, 3*nmax))
        #obtain species encoder

        spec_identity = tf.constant(self.species_identity, dtype=tf.int32) - 1

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

        elements = (batch_species_encoder, batch_width,
                positions, nmax_diff, batch_nats,
                batch_zeta, batch_width_ang, cells, replica_idx,
                batch_Rs_rad, batch_Rs_ang, batch_theta_s, C6)

        energies, forces = tf.map_fn(self.tf_predict_energy_forces, elements, fn_output_signature=[tf.float32, tf.float32])
        return energies, forces

    def force_loss(self, x):

        nat = tf.cast(x[0], tf.int32)
        force_ref = tf.reshape(x[1][:3*nat], (nat,3))
        force_pred = tf.reshape(x[2][:3*nat], (nat,3))

        loss = tf.reduce_mean((force_ref - force_pred)**2)

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

            loss = self.ecost * emse_loss
            loss += self.fcost * fmse_loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))


        metrics = {'tot_st': self._train_counter}
        mae = tf.reduce_mean(tf.abs(ediff / batch_nats))
        rmse = tf.sqrt(tf.reduce_mean((ediff/batch_nats)**2))

        mae_f = tf.map_fn(self.force_mae, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        mae_f = tf.reduce_mean(mae_f)

        rmse_f = tf.sqrt(fmse_loss)

        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})

        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})
        metrics.update({'loss': loss})
        metrics.update({'energy loss': emse_loss})
        metrics.update({'force loss': fmse_loss})
        lr = K.eval(self.optimizer.lr)

        #with writer.set_as_default():
        with self.train_writer.as_default(step=self._train_counter):

            tf.summary.scalar('1. Losses/1. Total',loss,self._train_counter)
            tf.summary.scalar('1. Losses/2. Energy',emse_loss,self._train_counter)
            tf.summary.scalar('1. Losses/3. Forces',fmse_loss,self._train_counter)
            tf.summary.scalar('2. Metrics/1. RMSE/atom',rmse,self._train_counter)
            tf.summary.scalar('2. Metrics/2. MAE/atom',mae,self._train_counter)
            tf.summary.scalar('2. Metrics/3. RMSE_F',rmse_f,self._train_counter)
            tf.summary.scalar('2. Metrics/3. MAE_F',mae_f,self._train_counter)
            tf.summary.histogram('3. Parameters/1. width',self.width_value,self._train_counter)
            tf.summary.histogram('3. Parameters/2. width_ang',self.width_value_ang,self._train_counter)
            tf.summary.histogram('3. Parameters/3. zeta',self.zeta_value,self._train_counter)
            tf.summary.histogram('3. Parameters/4. Rs_rad',self._Rs_rad,self._train_counter)
            tf.summary.histogram('3. Parameters/5. Rs_ang',self._Rs_ang,self._train_counter)
            tf.summary.histogram('3. Parameters/6. thetas',self._thetas,self._train_counter)
        return {key: metrics[key] for key in metrics.keys()}


    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs_target = data
        inputs = inputs_target[:6]
        target = inputs_target[6]

        e_pred, forces = self(inputs, training=True)  # Forward pass

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

        mae_f = tf.map_fn(self.force_mae, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
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
        inputs_target = data
        inputs = inputs_target[:6]
        target = inputs_target[6]
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

        mae_f = tf.map_fn(self.force_mae, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        mae_f = tf.reduce_mean(mae_f)

        rmse_f = tf.sqrt(fmse)

        metrics = {}

        metrics.update({'MAE': mae})
        metrics.update({'RMSE': rmse})
        metrics.update({'MAE_F': mae_f})
        metrics.update({'RMSE_F': rmse_f})

        return [target, e_pred, metrics, tf.reshape(target_f, [-1, nmax, 3]),tf.reshape(forces, [-1, nmax, 3]), batch_nats]
