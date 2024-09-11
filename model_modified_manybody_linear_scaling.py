from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import mendeleev
from mendeleev import element
import math 
import itertools, os
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from networks import Networks
import helping_functions as help_fn

class mBP_model(tf.keras.Model):
   
    def __init__(self, layer_sizes, rcut, species_identity, 
                 width, batch_size,
                activations,
                rc_ang,RsN_rad, RsN_ang,
                thetaN,width_ang,zeta,
                fcost=0.0,
                ecost=1.0,
                pbc=True,
                nelement=118,
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
                species_out_act='linear'):
        
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
        self.zeta = int(zeta)
        self.width_ang = float(width_ang)
        self.body_order = body_order
        
        #define learnable parameter per lxlylz components
        #this is the sum of all permutations = sum_{n=0}^zeta (n+2)^C_2
        i = tf.constant(0)
        res = tf.constant(0., dtype=tf.float32)
        cond = lambda i, res: tf.less(i, self.zeta+1)
        #compute (n+2)^C_2 = (n+2)!/n!/2!
        body = lambda i, res: [i+1, res+tf.cast(tf.reduce_prod(tf.range(1,i+3, dtype=tf.float32)) / (tf.reduce_prod(tf.range(1,i+1, dtype=tf.float32)) * 2), tf.float32)]
        n_perm = tf.while_loop(cond, body, [i, res])[1]
        n_perm = tf.cast(n_perm, tf.int32)
        self.n_perm = n_perm

        base_size = self.RsN_rad * 2 * self.n_perm

        self.feature_size = self.RsN_rad + base_size
        if self.body_order >= 4:
            for k in range(3,self.body_order):
                self.feature_size += base_size*self.n_perm
        self.nspec_embedding = nspec_embedding
        self.feature_size *= self.nspec_embedding
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
     
        self.atomic_nets = Networks(self.feature_size, self.layer_sizes, self._activations, l1=self.l1, l2=self.l2)

        # the number of elements in the periodic table
        self.nelement = nelement
        # create a species embedding network with 1 hidden layer Nembedding x Nspecies
        self.species_nets = Networks(self.nelement, 
                                     [self.nspec_embedding,1], 
                                     ['tanh',self.species_out_act], 
                                     #weight_initializer=init,
                                     #bias_initializer='zeros',
                                     prefix='species_encoder')
        #self.species_nets = Networks(self.nelement, [1], [self.species_out_act], prefix='species_encoder')

        constraint = None
        Nwidth_rad = self.RsN_rad
        Nwidth_ang = self.RsN_ang
        init = tf.keras.initializers.RandomNormal(mean=10, stddev=0.05)
        self.rbf_nets = Networks(1, [self.RsN_rad], ['sigmoid'], 
                                 weight_initializer=init,
                                 bias_initializer='zeros',
                                 prefix='rbf')
        init = tf.keras.initializers.RandomNormal(mean=10, stddev=0.05)
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
        init = tf.keras.initializers.GlorotNormal(seed=42)
        self.Rs_rad_nets = Networks(1, [self.RsN_rad], ['sigmoid'],
                                      weight_initializer=init,
                                      bias_initializer=init,
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='Rs_rad')
        init = tf.keras.initializers.GlorotNormal(seed=43)
        self.Rs_ang_nets = Networks(1, [self.RsN_ang], ['sigmoid'],
                                      weight_initializer=init,
                                      bias_initializer=init,
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='Rs_ang')
        init = tf.keras.initializers.GlorotNormal(seed=44)
        self.lambdas_nets = Networks(1, [self.thetaN], ['tanh'],
                                      weight_initializer=init,
                                      bias_initializer=init,
                                      kernel_constraint=constraint,
                                      bias_constraint=constraint, prefix='Rs_ang')
        '''
        #        self.n_perm = 20

        '''
        init = tf.keras.initializers.GlorotNormal(seed=45)
        self.weights_lxlylz_nets = Networks(1, [self.n_perm * self.n_perm], ['tanh'],
                                          weight_initializer=init,
                                          bias_initializer=init,
                                          kernel_constraint=constraint,
                                       bias_constraint=constraint, prefix='weights_lxlylz')
        '''
    @tf.function(
                input_signature=[(
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(3,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                )])
    def tf_predict_energy_forces(self,x):
        '''
        elements = (batch_species_encoder, batch_kn_rad,
                positions, nmax_diff, batch_nats,
                batch_kn_rad, cells, replica_idx,C6)
        '''
        rc = tf.cast(self.rcut,dtype=tf.float32)
        Ngauss = tf.cast(self.RsN_rad, dtype=tf.int32)
        rc_ang = tf.cast(self.rcut_ang, dtype=tf.float32)
        Ngauss_ang = tf.cast(self.RsN_ang, dtype=tf.int32)
        thetasN = tf.cast(self.thetaN, dtype=tf.int32)



        nat = x[4]
        nmax_diff = x[3]
        species_encoder = tf.reshape(x[0][:nat*self.nspec_embedding], [nat,self.nspec_embedding])
        kn_rad = tf.cast(x[1], dtype=tf.float32)
        positions = tf.reshape(x[2][:nat*3], [nat,3])
        #positions = positions

        #zeta cannot be a fraction
        zeta = tf.cast(self.zeta, dtype=tf.int32)
        kn_ang = tf.cast(x[5], dtype=tf.float32)
        
        cell = x[6]
        replica_idx = tf.cast(x[7], tf.int32)
        #Rs = x[8]
        #Rs_ang = x[9]
        #_lambda = tf.constant([-1.0, 1.0], tf.float32)
        #_lambda = x[10]

        #_weights_lxlylz = tf.reshape(x[11], [self.n_perm,self.n_perm])
        evdw = 0.0
        
        if self.include_vdw:
            C6 = tf.cast(x[8][:nat], tf.float32)

        with tf.GradientTape() as g:
            g.watch(positions)
            
            if self.pbc:
                
                #species_encoder0 = tf.identity(species_encoder)

                if self.include_vdw:
                    positions_extended, species_encoder, C6_extended = help_fn.generate_periodic_images(species_encoder, positions, cell, replica_idx, C6, self.include_vdw)
                    C6_extended = tf.reshape(C6_extended, [-1])
                else:
                    positions_extended, species_encoder = help_fn.generate_periodic_images(species_encoder, positions, cell, replica_idx)
                species_encoder_extended = tf.reshape(species_encoder_extended, [-1, self.nspec_embedding])
                positions_extended = tf.reshape(positions_extended, [-1, 3])
            else:
                positions_extended = tf.identity(positions)
                species_encoder_extended = tf.identity(species_encoder)
                #species_encoder0 = tf.identity(species_encoder)
                if self.include_vdw:
                    C6_extended = tf.identity(C6)
            #positions_extended has nat x nreplica x 3 for periodic systems and nat x 1 x 3 for molecules
            
            #rj-ri
            #Nneigh x nat x 3
            all_rij = positions - positions_extended[:, tf.newaxis, :]
            #nat x Nneigh x 3
            all_rij = tf.transpose(all_rij, [1,0,2])
            species_encoder_ij = tf.transpose(species_encoder * species_encoder_extended[:,tf.newaxis,:], [1,0,2]) #nat, nneigh, nembedding
            
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
                evdw = help_fn.vdw_contribution((all_rij_norm, C6ij))[0]
    #            tf.debugging.check_numerics(evdw, message='Total_energy_vdw contains NaN')

            inball_rad = tf.where(tf.logical_and(all_rij_norm <=rc, all_rij_norm>1e-8))
            all_rij_norm = tf.gather_nd(all_rij_norm, inball_rad)
            #produces as list of tensors with different shapes because atoms have different number of neighbors
            all_rij_norm = tf.RaggedTensor.from_value_rowids(all_rij_norm, inball_rad[:,0]).to_tensor(default_value=1e-8)
            
            all_rij = tf.gather_nd(all_rij, inball_rad)
            #produces as list of tensors with different shapes because atoms have different number of neighbors
            all_rij = tf.RaggedTensor.from_value_rowids(all_rij, inball_rad[:,0]).to_tensor(default_value=1e-8)
            species_encoder_ij = tf.gather_nd(species_encoder_ij, inball_rad)
            species_encoder_ij = tf.RaggedTensor.from_value_rowids(species_encoder_ij, inball_rad[:,0]).to_tensor()

            #nat x Ngauss x Nneigh
            #gauss_args = width[tf.newaxis,:,tf.newaxis] * (all_rij_norm[:,tf.newaxis,:] - Rs[tf.newaxis,:,tf.newaxis])**2

            #this is no needed because have no contribution from atoms outside cutoff because of fc
            #mask the non-zero values after ragged tensor is converted to fixed shape
            # This is very important to avoid adding contributions from atoms outside the sphere
            #nonzero_val = tf.where(all_rij_norm > 1e-8, 1.0,0.0)
            #nonzero_val = tf.cast(nonzero_val, tf.float32)
            #gauss_args = gauss_args * nonzero_val[:,tf.newaxis,:]



            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x 1 x Nneigh
            #fcuts is nat x Nneigh and reshaped to nat x 1 x Nneigh
            #args = species_encoder_rad[:,tf.newaxis,:] * help_fn.tf_app_gaussian(gauss_args) * help_fn.tf_fcut(all_rij_norm, rc)[:,tf.newaxis,:]
            
            # sum over neighbors j including periodic boundary conditions
            #atomic_descriptors = tf.reduce_sum(args, axis=-1)

            tf_pi = tf.constant(math.pi, dtype=tf.float32)
            arg = tf_pi / rc * tf.einsum('l,ij->ijl',tf.range(1, Ngauss+1, dtype=tf.float32) * kn_rad, all_rij_norm)
            arg = tf.reshape(arg, [-1])
            r = tf.einsum('l,ij->ijl',tf.ones(Ngauss), all_rij_norm)
            r = tf.reshape(r, [-1])
            bf_radial = tf.reshape(help_fn.bessel_function(r,arg,rc), [-1, Ngauss])


            radial_ij = tf.einsum('ij,il->ijl',bf_radial, tf.reshape(species_encoder_ij, [-1,self.nspec_embedding])) # shape=(na*nneigh,Ngauss*nspec)
            # sum over neighbors j including periodic boundary conditions
            radial_ij = tf.reshape(radial_ij, [nat,-1, self.nspec_embedding*Ngauss])
            atomic_descriptors = tf.reduce_sum(radial_ij, axis=1)
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, -1])





            #implement angular part: compute vect_rij dot vect_rik / rij / rik in a linear scaling form
            radial_ij = tf.reshape(radial_ij, [nat,-1,Ngauss*self.nspec_embedding])


            #expansion index
            n = tf.range(zeta+1, dtype=tf.int32)
            
            lxlylz = tf.map_fn(help_fn.find_three_non_negative_integers, n, 
                               fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32),
                               parallel_iterations=self.zeta+1)
            
            lxlylz = tf.reshape(lxlylz, [-1,3])
            lxlylz_sum = tf.reduce_sum(lxlylz, axis=-1) # the value of n for each lx, ly and lz

            reg = 1e-20
            all_rij_norm_inv = 1.0 / (all_rij_norm + reg)
            #tf.debugging.check_numerics(all_rij_norm_inv, message='all_rij_norm_inv contains NaN')

            rij_unit = tf.einsum('ijk,ij->ijk',all_rij, all_rij_norm_inv)

            #rx^lx * ry^ly * rz^lz
            #this need to be regularized to avoid undefined derivatives
            rij_lxlylz = (rij_unit[:,:,tf.newaxis,:] + 1e-16)**(tf.cast(lxlylz, tf.float32)[tf.newaxis,tf.newaxis,:,:])
            g_ij_lxlylz = tf.reduce_prod(rij_lxlylz, axis=-1) #nat x neigh x n_lxlylz
#            tf.debugging.check_numerics(g_ij_lxlylz, message='1. g_ij_lxlylz contains NaN')
            
            #compute normalizations n! / lx!/ly!/lz!
            nfact = tf.map_fn(help_fn.factorial, lxlylz_sum, fn_output_signature=tf.float32,
                              parallel_iterations=4) #computed for all n_lxlylz

            #lx!ly!lz!
            fact_lxlylz = tf.reshape(tf.map_fn(help_fn.factorial, tf.reshape(lxlylz, [-1]), fn_output_signature=tf.float32,
                                               parallel_iterations=4), [-1,3])
            fact_lxlylz = tf.reduce_prod(fact_lxlylz, axis=-1)

            nfact_lxlylz = nfact / fact_lxlylz # n_lxlylz
            
            #compute zeta! / (zeta-n)! / n!

            zeta_fact = help_fn.factorial(zeta)
            zeta_fact_n = tf.map_fn(help_fn.factorial, zeta-lxlylz_sum, 
                                    fn_output_signature=tf.float32,
                                    parallel_iterations=4)

            zetan_fact = zeta_fact / (zeta_fact_n * nfact)

            fact_norm = nfact_lxlylz * zetan_fact

            g_ij_lxlylz = tf.einsum('ijk,k->ijk',g_ij_lxlylz, fact_norm) # shape=(nat, neigh, n_lxlylz)

            #mix all angular momentum components to improve functional flexibility
            #g_ij_lxlylz = tf.squeeze(tf.matmul(_weights_lxlylz, g_ij_lxlylz[:,:,:,tf.newaxis])) # nat, neigh, n_lxlylz
 #           tf.debugging.check_numerics(g_ij_lxlylz, message='g_ij_lxlylz contains NaN')
           
            #sum over neighbors = nat x nrs x nl
            g_ilxlylz = tf.einsum('ijk,ijl->ikl',radial_ij, g_ij_lxlylz) #shape=(nat,nrad,n_lxlylz)

  #          tf.debugging.check_numerics(g_ilxlylz, message='g_ilxlylz contains NaN')

            #for the sum over k, we just square the gs to give 3 body terms
            g3_ilxlylz = g_ilxlylz * g_ilxlylz
   #         tf.debugging.check_numerics(g3_ilxlylz, message='g3_ilxlylz contains NaN')

            
            #g3_ilxlylz *= fact_norm[tf.newaxis,tf.newaxis,:]

            #compute lambda ^ n

            lambda_n = tf.constant([-1.0, 1.0], dtype=tf.float32)[:,tf.newaxis] ** tf.cast(lxlylz_sum, tf.float32) # shape=(2,n_lxlylz)
            lambda_n = tf.reshape(lambda_n, [2, -1])
#            lambda_n =(_lambda[:,tf.newaxis] + 1e-16) ** tf.cast(lxlylz_sum, tf.float32) #n_lambda x n_lxlylz
            lambda_norm = 2.0
            #lambda_norm = 1. + tf.abs(_lambda) # (n_lambda,)
            #lambda2 = _lambda * _lambda
            #lambda_norm = 1. + tf.sqrt(lambda2 + 1e-16) # (n_lambda,)
            lambda_n /= (lambda_norm ** tf.cast(zeta,tf.float32))
            #tf.debugging.check_numerics(lambda_n, message='lambda_n contains NaN')

            g3_ilxlylz = tf.einsum('ijk,lk->ijlk', g3_ilxlylz, lambda_n) #(nat,nrs,n_lambda, n_lxlylz)
            g3_i = tf.reshape(g3_ilxlylz, [nat,-1]) #(nat,nrs*n_lambda*n_lxlylz) after sum over n and lxlylz
            #g3_i = tf.reduce_sum(g3_i, axis=-1) #(nat,nrs,n_lambda) after sum over n and lxlylz

            #body_descriptor_3 = tf.reshape(g3_i, [nat, Ngauss_ang*thetasN])
            atomic_descriptors = tf.concat([atomic_descriptors, g3_i], axis=1)

            if self.body_order >= 4:
                #compute sum over k and l
                g_i34_lm = tf.einsum('ijk,ijl->ijkl',g_ilxlylz, g_ilxlylz) #nat x nrs x nl x nl
#                tf.debugging.check_numerics(g_i34_lm, message='g_ij_lxlylz contains NaN')
                g_ij_lm = tf.einsum('ijk,ijl->ijkl', g_ij_lxlylz, g_ij_lxlylz) # shape=(nat,neigh, nl, nl)
                gi_2_lm = tf.einsum('ijk,ijlm->iklm',radial_ij, g_ij_lm) # shape=(nat, nrs, nl,nl)
                   
                g4_i_lm = tf.reshape(gi_2_lm * g_i34_lm, [nat,-1, self.n_perm*self.n_perm]) #nat x nrs x nl**2
                #coefficients
#                fact_norm_lm = fact_norm[:,tf.newaxis] * fact_norm[tf.newaxis,:] #nl x nl
                lambda_n_lm = tf.reshape(tf.einsum('ij,ik->ijk',lambda_n, lambda_n), [2,-1]) # nlambda x nl x nl
                #tf.debugging.check_numerics(lambda_n_lm, message='lambda_lm contains NaN')
                g4_i = tf.einsum('ijk,lk->ijlk',g4_i_lm, lambda_n_lm) # shape=(nat, nrs, nlambda=2,nl**2)
 #               tf.debugging.check_numerics(g4_i, message='g4_i contains NaN')

                body_descriptor_4 = tf.reshape(g4_i, [nat, -1])
                atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_4], axis=1)


            #four body descriptors 
#            lxlylz_idx = tf.map(tf.range, tf.reshape(lxlylz, [-1]))

            '''lxlylz_4 = tf.map_fn(self.get_all_lxyz_idx, t, fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32)) 
            lxlylz_4 = tf.reshape(lxlylz_4, [-1,3])

            #lx = tf.map_fn(tf.range, lxlylz_4[:,0]+1, fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32))
            #ly = tf.map_fn(tf.range, lxlylz_4[:,1]+1, fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32))
            #lz = tf.map_fn(tf.range, lxlylz_4[:,2]+1, fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32))
            lx = lxlylz_4[:,0]
            ly = lxlylz_4[:,1]
            lz = lxlylz_4[:,2]
            
            # one per term in the sum
            lxyz = tf.map_fn(self.generate_all_lxyz, , fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32)) 
            lxyz = tf.reshape(idx, [-1,3])

            #lx_rep = tf.map_fn(self.help_func, lxlylz[:,0], fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32))
            #lx_rep = tf.reshape(lx_rep, [-1])
            lx_ab = lxyz[:,0] - lx + ly
            lx_ac = lxyz[:,0] - lx + lz
            
            #ly_rep = tf.map_fn(self.help_func, lxlylz[:,1], fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32))
            #ly_rep = tf.reshape(ly_rep, [-1])
            ly_ba = lxyz[:,1] - ly + lx
            ly_bc = lxyz[:,1] - ly + lz
            
            #lz_rep = tf.map_fn(self.help_func, lxlylz[:,2], fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32))
            #lz_rep = tf.reshape(lz_rep, [-1])
            lz_ca = lxyz[:,2] - lz + lx
            lz_cb = lxyz[:,2] - lz + ly
            #next, compute lxyz[:,0]! / (lxyz[:,0]-lx)! / lx! and also for y and z components 

#            lxlylz_sum = 
            '''

            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])
            
            #predict energy and forces

            atomic_energies = self.atomic_nets(atomic_descriptors)

            total_energy = tf.reduce_sum(atomic_energies)
            

            if self.include_vdw:
                total_energy += evdw

        forces = g.gradient(total_energy, positions)
        
        padding = tf.zeros((nmax_diff,3))
        forces = tf.concat([-forces, padding], 0)
        #forces = tf.zeros((nmax_diff+nat,3))   
        #return [tf.cast(total_energy, tf.float32), -tf.cast(forces, tf.float32)]
        return [total_energy, forces]

    def call(self, inputs, training=False):
        '''inpu has a shape of batch_size x nmax_atoms x feature_size'''
        # may be just call the energy prediction here which will be needed in the train and test steps
        # the input are going to be filename from which descriptors and targets are going to be extracted

        batch_size = tf.shape(inputs[2])[0]

        inputs_width = tf.ones(1)
        self.kn_rad = tf.reshape(self.rbf_nets(inputs_width[tf.newaxis, :]), [-1])
        self.kn_ang = tf.reshape(self.rbf_nets_ang(inputs_width[tf.newaxis, :]), [-1])


        #inputs_width = tf.ones(1)
        #self.width_value = tf.reshape(self.width_nets(inputs_width[tf.newaxis, :]), [-1])

        #inputs_width_ang = tf.ones(1)
        #self.width_value_ang = tf.reshape(self.width_nets_ang(inputs_width_ang[tf.newaxis, :]), [-1])

                
        #inputs for center networks
        tf_pi = tf.constant(math.pi, dtype=tf.float32)
        #Rs = tf.ones(1)
        #Rs_ang = tf.ones(1)
        #_lambdas = tf.ones(1)
        #Rs_rad_pred = tf.reshape(self.Rs_rad_nets(Rs[tf.newaxis,:]), [-1])
        #self._Rs_rad = help_fn.rescale_params(Rs_rad_pred, self.min_radial_center, self.rcut)

        #Rs_ang_pred = tf.reshape(self.Rs_ang_nets(Rs_ang[tf.newaxis,:]), [-1])
        #self._Rs_ang = help_fn.rescale_params(Rs_ang_pred, self.min_radial_center, self.rcut_ang)
        
        #self.lambdas = tf.reshape(self.lambdas_nets(_lambdas[tf.newaxis,:]), [-1])
        #self.lambdas = help_fn.rescale_params(lambdas, -1.0, 1.0)

        
        
        #self.weights_lxlylz = tf.reshape(self.weights_lxlylz_nets(tf.ones(1)[tf.newaxis,:]), [-1])
        batch_kn_rad = tf.tile([self.kn_rad], [batch_size,1])
        batch_kn_ang = tf.tile([self.kn_ang], [batch_size,1])

        #batch_width = tf.tile([self.width_value], [batch_size,1])
        #batch_width_ang = tf.tile([self.width_value_ang], [batch_size,1])
        #batch_Rs_rad = tf.tile([self._Rs_rad], [batch_size,1])
        #batch_Rs_ang = tf.tile([self._Rs_ang], [batch_size,1])
        #batch_lambdas = tf.tile([self.lambdas], [batch_size,1])
        #batch_weights_lxlylz = tf.tile([self.weights_lxlylz], [batch_size,1])


        batch_nats = inputs[2]
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)
        batch_nmax = tf.tile([nmax], [batch_size])
        nmax_diff = batch_nmax - batch_nats

        #positions and species_encoder are ragged tensors are converted to tensors before using them
        positions = tf.reshape(inputs[0].to_tensor(shape=(-1,nmax,3)), (-1, 3*nmax))
        #obtain species encoder

        spec_identity = tf.constant(self.species_identity, dtype=tf.int32) - 1

        species_one_hot_encoder = tf.one_hot(spec_identity, depth=self.nelement)

        self.trainable_species_encoder = self.species_nets(species_one_hot_encoder)
        species_encoder = tf.cast(inputs[1], tf.float32).to_tensor(shape=(-1, nmax))
        batch_species_encoder = tf.zeros([batch_size, nmax, self.nspec_embedding], dtype=tf.float32)
        # This may be implemented better but not sure how yet
        for idx, spec in enumerate(self.species_identity):
            values = tf.ones([batch_size, nmax, self.nspec_embedding], dtype=tf.float32) * self.trainable_species_encoder[idx]
            batch_species_encoder += tf.where(tf.equal(tf.tile(species_encoder[:,:,tf.newaxis], [1,1,self.nspec_embedding]),
                                                       tf.cast(spec,tf.float32)),
                    values, tf.zeros([batch_size, nmax, self.nspec_embedding]))
        batch_species_encoder = tf.reshape(batch_species_encoder, [-1,self.nspec_embedding*nmax])

        cells = inputs[3]
        replica_idx = inputs[4]
        C6 = inputs[5]

        elements = (batch_species_encoder, batch_kn_rad,
                positions, nmax_diff, batch_nats,
                batch_kn_ang, cells, replica_idx,C6)
                #batch_Rs_rad, batch_Rs_ang, batch_lambdas, C6)

        energies, forces = tf.map_fn(self.tf_predict_energy_forces, elements, fn_output_signature=[tf.float32, tf.float32],
                                     parallel_iterations=self.batch_size)
        return energies, forces

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs_target = data
        inputs = inputs_target[:6]
        target = tf.cast(inputs_target[6], tf.float32)

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

            fmse_loss = tf.map_fn(help_fn.force_loss, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
            fmse_loss = tf.reduce_mean(fmse_loss)

            loss = self.ecost * emse_loss
            loss += self.fcost * fmse_loss
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))


        #metrics = {'tot_st': self._train_counter}
        #metrics = {'tot_st': self.step}
        metrics = {}
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
        metrics.update({'energy loss': emse_loss})
        metrics.update({'force loss': fmse_loss})

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

            tf.summary.histogram('3. Parameters/1. kn_rad',self.kn_rad,self._train_counter)
            tf.summary.histogram('3. Parameters/2. kn_ang',self.kn_ang,self._train_counter)

            #tf.summary.histogram('3. Parameters/1. width',self.width_value,self._train_counter)
            #tf.summary.histogram('3. Parameters/2. width_ang',self.width_value_ang,self._train_counter)
            #tf.summary.histogram('3. Parameters/3. Rs_rad',self._Rs_rad,self._train_counter)
            #tf.summary.histogram('3. Parameters/4. Rs_ang',self._Rs_ang,self._train_counter)
            #tf.summary.histogram('3. Parameters/5. lambda',self.lambdas,self._train_counter)
            #tf.summary.histogram('3. Parameters/6. weights_lxlylz',self.weights_lxlylz,self._train_counter)
        return {key: metrics[key] for key in metrics.keys()}


    def test_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        inputs_target = data
        inputs = inputs_target[:6]
        target = tf.cast(inputs_target[6], tf.float32)

        e_pred, forces = self(inputs, training=True)  # Forward pass

        batch_nats = tf.cast(inputs[2], tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)

        forces = tf.reshape(forces, [-1, nmax*3])

        target_f = tf.reshape(inputs_target[7].to_tensor(), [-1, 3*nmax])
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
        inputs_target = data
        inputs = inputs_target[:6]
        target = tf.cast(inputs_target[6], tf.float32)
        e_pred, forces = self(inputs, training=False)  # Forward pass

        batch_nats = tf.cast(inputs[2], tf.float32)
        nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)

        forces = tf.reshape(forces, [-1, nmax*3])

        target_f = tf.reshape(inputs_target[7].to_tensor(), [-1, 3*nmax])
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

