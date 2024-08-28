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
                fcost=0.0,
                ecost=1.0,
                pbc=True,
                nelement=118,
                train_writer=None,
                l1=0.0,l2=0.0,
                nspec_embedding=64,
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
        
        base_size = self.RsN_ang * self.thetaN * 2
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

        init = tf.keras.initializers.GlorotNormal(seed=34569)
        self.weights_lxlylz_nets = Networks(1, [self.thetaN * n_perm], ['sigmoid'],
                                          weight_initializer=init,
                                          bias_initializer=init,
                                          kernel_constraint=constraint,
                                          bias_constraint=constraint, prefix='weights_lxlylz')


    #start functions from chatgpt
    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32),])
    def find_three_non_negative_integers(self, n):
        # Create a tensor of integers from 0 to n
        i = tf.range(0, n + 1)
        j = tf.range(0, n + 1)
        
        # Create a meshgrid for all combinations of i and j
        I, J = tf.meshgrid(i, j, indexing='ij')
        
        # Calculate k based on the sum condition
        K = n - I - J
        
        # Create a mask to filter out invalid (negative) k values
        valid_mask = tf.greater_equal(K, 0)

        # Use the mask to extract valid triplets
        valid_i = tf.boolean_mask(I, valid_mask)
        valid_j = tf.boolean_mask(J, valid_mask)
        valid_k = tf.boolean_mask(K, valid_mask)
        
        # Stack valid triplets together
        valid_triplets = tf.stack([valid_i, valid_j, valid_k], axis=1)
        
        return tf.reshape(valid_triplets, [-1])
         
    @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32),])
    def factorial(self, n):
        # Create a tensor of integers from 1 to n
        # Use tf.range to create a sequence from 1 to n + 1
        numbers = tf.range(1, n + 1, dtype=tf.float32)

        # Calculate the factorial using tf.reduce_prod to multiply all elements
        result = tf.reduce_prod(numbers)

        return result
    def generate_periodic_images(self, species_vectors, positions, lattice_vectors, image_range, C6=None):
        """
        Generate periodic image points for given atomic positions with a cutoff distance.

        Parameters:
            positions (tf.Tensor): Tensor of atomic positions in Cartesian coordinates (shape: (n_atoms, 3)).
            lattice_vectors (tf.Tensor): Tensor of lattice vectors (shape: (3, 3)).
            image_range (int): The range of periodic images to generate in each direction.
            cutoff (float): Cutoff distance to limit generated images.

        Returns:
            tf.Tensor: Tensor of periodic image positions within the cutoff distance.
        """
        # Create meshgrid for integer translations
        translations_x = tf.range(-image_range[0], image_range[0] + 1)
        translations_y = tf.range(-image_range[1], image_range[1] + 1)
        translations_z = tf.range(-image_range[2], image_range[2] + 1)
        tx, ty, tz = tf.meshgrid(translations_x, translations_y, translations_z, indexing='ij')
        
        # Stack translations to create a list of all translation vectors
        translation_vectors = tf.stack([tx, ty, tz], axis=-1)  # Shape: (image_range * 2 + 1, image_range * 2 + 1, image_range * 2 + 1, 3)
        
        # Reshape translation vectors for broadcasting
        translation_vectors = tf.reshape(translation_vectors, [-1, 3])  # Shape: ((image_range * 2 + 1)^3, 3)

        # Repeat positions for each atom
        atom_positions = tf.expand_dims(positions, axis=1)  # Shape: (n_atoms, 1, 3)
        atom_positions = tf.repeat(atom_positions, tf.shape(translation_vectors)[0], axis=1)  # Shape: (n_atoms, (image_range * 2 + 1)^3, 3)
        
        
        # Generate periodic images
        # Expand the translation_vectors for proper broadcasting
        expanded_translations = tf.expand_dims(translation_vectors, axis=0)  # Shape: (1, (image_range * 2 + 1)^3, 3)
        
        # Perform the addition
        periodic_images = atom_positions + tf.tensordot(tf.cast(expanded_translations,tf.float32), lattice_vectors, axes=[2, 0])  # Shape: (n_atoms, (image_range * 2 + 1)^3, 3)
        
        species_vectors = tf.expand_dims(species_vectors, axis=1)
        species_vectors = tf.repeat(species_vectors, tf.shape(translation_vectors)[0], axis=1)

        if self.include_vdw:
            C6_extended = tf.expand_dims(C6, axis=1)
            C6_extended = tf.repeat(C6_extended, tf.shape(translation_vectors)[0], axis=1)
            return periodic_images, species_vectors, C6_extended

        # Calculate distances from the original positions
        #distances = periodic_images - tf.expand_dims(positions, axis=1), axis=-1)  # Shape: (n_atoms, (image_range * 2 + 1)^3)

        # Apply the cutoff to filter out positions
        #valid_images = tf.boolean_mask(periodic_images, distances < cutoff)  # Shape: (n_valid_images, 3)

        return periodic_images, species_vectors


    def calculate_image_range_per_vector(self, cutoff, lattice_vectors):
        """
        Calculate the image range for each lattice vector based on a cutoff distance.

        Parameters:
            cutoff (float): The cutoff distance for interactions.
            lattice_vectors (tf.Tensor): Tensor of lattice vectors (shape: (3, 3)).

        Returns:
            tf.Tensor: A tensor containing the determined image ranges for each lattice vector.
        """
        # Calculate the lengths of the lattice vectors
        lattice_lengths = tf.norm(lattice_vectors, axis=1)  # Shape: (3,)

        # Calculate the image range for each lattice vector
        image_ranges = tf.floor(cutoff / lattice_lengths)

        return tf.cast(image_ranges, tf.int32)
    #end chatgpt sections

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

    @tf.function(input_signature=[tf.TensorSpec(shape=(None,tf.newaxis), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.float32)])
    def tf_fcut(self,r,rc):
        dim = tf.shape(r)
        pi = tf.constant(math.pi, dtype=tf.float32)
        return tf.where(tf.logical_and(r<=rc,r>1e-8), 0.5*(1.0 + tf.cos(pi*r/rc)), tf.zeros(dim, dtype=tf.float32))
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,tf.newaxis), dtype=tf.float32),
                                 tf.TensorSpec(shape=(), dtype=tf.float32)])
    def _tf_fcut(self,r,rc):
        dim = tf.shape(r)
        #pi = tf.constant(math.pi, dtype=tf.float32)
        x = tf.tanh(1 - r / rc)
        return tf.where(tf.logical_and(r<=rc,r>1e-8), x*x*x, tf.zeros(dim, dtype=tf.float32))
    @tf.function(input_signature=[tf.TensorSpec(shape=(None,tf.newaxis,tf.newaxis), dtype=tf.float32)])
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

    def help_func(self, n):
        return tf.ones(n+1, tf.int32) * n
    def get_all_lxyz_idx(self, n):
        '''all terms in the lx,lylz summation for the cross product'''
        i = tf.range(0, n[0] + 1)
        j = tf.range(0, n[1] + 1)
        k = tf.range(0, n[2] + 1)
        X = tf.meshgrid(i,j,k, indexing='ij')
        return tf.reshape(tf.transpose(X), [-1])

    def generate_all_lxyz(self, n):
        '''terms needed to compute the lxyz factorial'''
        _n = tf.reduce_prod(n+1)
        v = tf.repeat(tf.expand_dims(n, axis=0), _n, axis=0)
        return tf.reshape(v, [-1])

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
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
#                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                )])
                #elements = (batch_species_encoder, batch_width,
                #positions, nmax_diff, batch_nats,
                #batch_width_ang, cells, replica_idx,
                #batch_Rs_rad, batch_Rs_ang, batch_lambdas, weights_lxlylz, C6)
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
        zeta = tf.cast(self.zeta, dtype=tf.int32)
        width_ang = tf.cast(x[5], dtype=tf.float32)
        
        cell = tf.cast(x[6], tf.float32)
        replica_idx = tf.cast(x[7], tf.float32)
        Rs = tf.cast(x[8], tf.float32)
        Rs_ang = tf.cast(x[9], tf.float32)
        _lambda = tf.constant([-1.0, 1.0], tf.float32)

        _weights_lxlylz = tf.reshape(x[10], [self.thetaN,self.n_perm])
        evdw = 0.0
        
        if self.include_vdw:
            C6 = tf.cast(x[11], tf.float32)

        with tf.GradientTape() as g:
            g.watch(positions)
            
            if self.pbc:
                
                species_encoder0 = tf.identity(species_encoder)

                if self.include_vdw:
                    positions_extended, species_encoder, C6_extended = generate_periodic_images(self, species_encoder, positions, lattice_vectors, replica_idx, C6)
                    C6_extended = tf.reshape(C6_extended, [-1])
                else:
                    positions_extended, species_encoder = generate_periodic_images(self, species_encoder, positions, lattice_vectors, replica_idx)
                species_encoder = tf.reshape(species_encoder, [-1])
                positions_extended = tf.reshape(positions_extended, [-1, 3])
            else:
                positions_extended = tf.identity(positions)
                species_encoder0 = tf.identity(species_encoder)
                if self.include_vdw:
                    C6_extended = tf.identity(C6)
            #positions_extended has nat x nreplica x 3 for periodic systems and nat x 1 x 3 for molecules
            
            #rj-ri
            #Nneigh x nat x 3
            all_rij = positions - positions_extended[:, tf.newaxis, :]
            #nat x Nneigh x 3
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


            #implement angular part: compute vect_rij dot vect_rik / rij / rik in a linear scaling form
            gauss_ang_args = width_ang[tf.newaxis,:,tf.newaxis] * (all_rij_norm[:,tf.newaxis,:] - Rs_ang[tf.newaxis,:,tf.newaxis])**2
            #nat x nrs x neigh
            Radial_ij =  species_encoder_rad[:,tf.newaxis,:] * self.tf_app_gaussian(gauss_ang_args) * self.tf_fcut(all_rij_norm, rc_ang)[:,tf.newaxis,:]

            #expansion index
            n = tf.range(zeta+1, dtype=tf.int32)
            
            lxlylz = tf.map_fn(self.find_three_non_negative_integers, n, 
                               fn_output_signature=tf.RaggedTensorSpec(shape=(None,),dtype=tf.int32))
            
            lxlylz = tf.reshape(lxlylz, [-1,3])
            lxlylz_sum = tf.reduce_sum(lxlylz, axis=-1) # the value of n for each lx, ly and lz

            reg = 1e-20
            all_rij_norm_inv = 1.0 / (all_rij_norm + reg)
            #tf.debugging.check_numerics(all_rij_norm_inv, message='all_rij_norm_inv contains NaN')

            rij_unit = all_rij * all_rij_norm_inv[:,:,tf.newaxis]

            #rx^lx * ry^ly * rz^lz
            #this need to be regularized to avoid undefined derivatives
            rij_lxlylz = (rij_unit[:,:,tf.newaxis,:] + 1e-20)**(tf.cast(lxlylz, tf.float32)[tf.newaxis,tf.newaxis,:,:])
            g_ilxlylz = tf.reduce_prod(rij_lxlylz, axis=-1) #nat x neigh x n_lxlylz
           
            #sum over neighbors
            g_ilxlylz = tf.reduce_sum(Radial_ij[:,:,:,tf.newaxis] * g_ilxlylz[:,tf.newaxis,:,:], axis=2)

            #for the sum over k, we just square the gs to give 3 body terms
            g3_ilxlylz = g_ilxlylz * g_ilxlylz

            #compute normalizations n! / lx!/ly!/lz!
            nfact = tf.map_fn(self.factorial, lxlylz_sum, fn_output_signature=tf.float32) #computed for all n_lxlylz

            #lx!ly!lz!
            fact_lxlylz = tf.reshape(tf.map_fn(self.factorial, tf.reshape(lxlylz, [-1]), fn_output_signature=tf.float32), [-1,3])
            fact_lxlylz = tf.reduce_prod(fact_lxlylz, axis=-1)

            nfact_lxlylz = nfact / fact_lxlylz # n_lxlylz
            
            #compute zeta! / (zeta-n)! / n!

            zeta_fact = self.factorial(zeta)
            zeta_fact_n = tf.map_fn(self.factorial, zeta-lxlylz_sum, fn_output_signature=tf.float32)

            zetan_fact = zeta_fact / (zeta_fact_n * nfact)

            fact_norm = nfact_lxlylz * zetan_fact

            g3_ilxlylz *= fact_norm[tf.newaxis,tf.newaxis,:]

            #compute lambda ^ n
            lambda_n = _lambda[:,tf.newaxis] ** tf.cast(lxlylz_sum, tf.float32) #n_lambda x n_lxlylz
            lambda_norm = 2.0 # (n_lambda,)

            g3_ilxlylz = g3_ilxlylz[:,:,tf.newaxis,:] * lambda_n[tf.newaxis,tf.newaxis,:,:] #(nat,nrs,2, n_lxlylz)
            g3_ilxlylz /=  lambda_norm ** tf.cast(zeta,tf.float32)

            #mix all angular momentum components to improve functional flexibility
            
            #natxnrsxnlambdaxn_lxlylz = n_lxlylz x n_lxlylz x nat x nrs x nlambda x n_lxlylz
            #each lxlylz components is multiplied by a learnable parameter
            #g3_ilxlylz =  tf.squeeze(tf.einsum('...jk,...kl->...jl', _weights_lxlylz, g3_ilxlylz[:,:,:,:,tf.newaxis]))
            #same result can be achieved by tf.squeeze(tf.matmul(_weights_lxlylz, g3_ilxlylz[:,:,:,:,tf.newaxis]))

            #g3_ilxlylz =  tf.squeeze(tf.einsum('...jk,...kl->...jl', _weights_lxlylz, g3_ilxlylz[:,:,:,:,tf.newaxis], optimize='auto')) # shape = natxnrsxnlambdaxn_lxlylz
            #shape = tf.shape(g3_ilxlylz)
            g3_i = tf.squeeze(tf.matmul(_weights_lxlylz, g3_ilxlylz[:,:,:,:,tf.newaxis])) # nat, nrs, 2, ntheta
            #g3_ilxlylz = tf.reshape(tf.matmul(_weights_lxlylz, g3_ilxlylz[:,:,:,:,tf.newaxis]), shape)
#            g3_i = tf.reduce_sum(g3_ilxlylz, axis=-1) #(nat,nrs,n_lambda) after sum over n and lxlylz

            body_descriptor_3 = tf.reshape(g3_i, [nat, 2*Ngauss_ang*thetasN])
            atomic_descriptors = tf.concat([atomic_descriptors, body_descriptor_3], axis=1)

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
            
            #mutiply by species weights per atom
            atomic_descriptors = species_encoder0[:,tf.newaxis] * atomic_descriptors
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
    def rescale_params(self, x, a, b):
        rsmin = tf.reduce_min(x)
        rsmax = tf.reduce_max(x)
        return a + (b - a) * (x - rsmin) / (rsmax - rsmin + 1e-12)

    def call(self, inputs, training=False):
        '''inpu has a shape of batch_size x nmax_atoms x feature_size'''
        # may be just call the energy prediction here which will be needed in the train and test steps
        # the input are going to be filename from which descriptors and targets are going to be extracted

        batch_size = tf.shape(inputs[2])[0]

        inputs_width = tf.ones(1)
        self.width_value = tf.reshape(self.width_nets(inputs_width[tf.newaxis, :]), [-1])

        inputs_width_ang = tf.ones(1)
        self.width_value_ang = tf.reshape(self.width_nets_ang(inputs_width_ang[tf.newaxis, :]), [-1])

                
        #inputs for center networks
        tf_pi = tf.constant(math.pi, dtype=tf.float32)
        Rs = tf.ones(1)
        Rs_ang = tf.ones(1)
        _lambdas = tf.ones(1)
        Rs_rad_pred = tf.reshape(self.Rs_rad_nets(Rs[tf.newaxis,:]), [-1])
        self._Rs_rad = self.rescale_params(Rs_rad_pred, self.min_radial_center, self.rcut)

        Rs_ang_pred = tf.reshape(self.Rs_ang_nets(Rs_ang[tf.newaxis,:]), [-1])
        self._Rs_ang = self.rescale_params(Rs_ang_pred, self.min_radial_center, self.rcut_ang)
        
    #    self.lambdas = tf.reshape(self.lambdas_nets(_lambdas[tf.newaxis,:]), [-1])
        
        
        self.weights_lxlylz = tf.reshape(self.weights_lxlylz_nets(tf.ones(1)[tf.newaxis,:]), [-1])

        batch_width = tf.tile([self.width_value], [batch_size,1])
        batch_width_ang = tf.tile([self.width_value_ang], [batch_size,1])
        batch_Rs_rad = tf.tile([self._Rs_rad], [batch_size,1])
        batch_Rs_ang = tf.tile([self._Rs_ang], [batch_size,1])
     #   batch_lambdas = tf.tile([self.lambdas], [batch_size,1])
        batch_weights_lxlylz = tf.tile([self.weights_lxlylz], [batch_size,1])


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
        species_encoder = inputs[1].to_tensor(shape=(-1, nmax))
        batch_species_encoder = tf.zeros([batch_size, nmax], dtype=tf.float32)

        for idx, spec in enumerate(self.species_identity):
            values = tf.ones([batch_size, nmax], dtype=tf.float32) * self.trainable_species_encoder[idx]
            batch_species_encoder += tf.where(tf.equal(species_encoder,tf.cast(spec,tf.float32)),
                    values, tf.zeros([batch_size, nmax]))
            
        cells = inputs[3]
        replica_idx = inputs[4]
        C6 = inputs[5]

        elements = (batch_species_encoder, batch_width,
                positions, nmax_diff, batch_nats,
                batch_width_ang, cells, replica_idx,
                batch_Rs_rad, batch_Rs_ang, batch_weights_lxlylz, C6)
                #batch_Rs_rad, batch_Rs_ang, batch_lambdas, batch_weights_lxlylz, C6)

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
            tf.summary.scalar('2. Metrics/4. MAE_F',mae_f,self._train_counter)
            for idx, spec in enumerate(self.species_identity):
                tf.summary.scalar(f'3. encoding /{idx}',self.trainable_species_encoder[idx][0],self._train_counter)

            tf.summary.histogram('3. Parameters/1. width',self.width_value,self._train_counter)
            tf.summary.histogram('3. Parameters/2. width_ang',self.width_value_ang,self._train_counter)
            tf.summary.histogram('3. Parameters/3. Rs_rad',self._Rs_rad,self._train_counter)
            tf.summary.histogram('3. Parameters/4. Rs_ang',self._Rs_ang,self._train_counter)
            #tf.summary.histogram('3. Parameters/5. lambda',self.lambdas,self._train_counter)
            tf.summary.histogram('3. Parameters/5. weights_lxlylz',self.weights_lxlylz,self._train_counter)
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
