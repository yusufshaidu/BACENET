from data_processing import data_preparation
from model_modified_manybody import mBP_model

def compute_features():
    '''input has a shape of batch_size x nmax_atoms x feature_size'''
    # may be just call the energy prediction here which will be needed in the train and test steps
    # the input are going to be filename from which descriptors and targets are going to be extracted

    batch_size = tf.shape(inputs[2])[0] 
    # the batch size may be different from the set batchsize saved in varaible self.batch_size
    # because the number of data point may not be exactly divisible by the self.batch_size.

    inputs_width = tf.ones(1)
    self.kn_rad = tf.reshape(self.rbf_nets(inputs_width[tf.newaxis, :]), [-1])
    self.kn_ang = tf.reshape(self.rbf_nets_ang(inputs_width[tf.newaxis, :]), [-1])
    #self.width_value = tf.reshape(self.width_nets(inputs_width[tf.newaxis, :]), [-1])

    #inputs_width_ang = tf.ones(1)
    #self.width_value_ang = tf.reshape(self.width_nets_ang(inputs_width_ang[tf.newaxis, :]), [-1])
    inputs_zeta = tf.ones(1)
    self.zeta_value = tf.reshape(self.zeta_nets(inputs_zeta[tf.newaxis, :]), [-1])
            
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
    species_encoder = inputs[1].to_tensor(shape=(-1, nmax)) #contains atomic number per atoms for all element in a batch
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
            batch_zeta, batch_kn_ang, cells, replica_idx,
            batch_theta_s, C6)

    features = tf.map_fn(self.tf_predict_energy_forces, elements, 
                                 fn_output_signature=[tf.float32, tf.float32],
                                 parallel_iterations=self.batch_size)
    return energies, forces
