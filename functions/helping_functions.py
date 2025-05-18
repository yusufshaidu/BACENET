import numpy as np
import tensorflow as tf
import math 

def find_three_non_negative_integers_handcoded(n):
    # Create a tensor of integers from 0 to n
    if n == 0:
        return [[0, 0, 0]]
    if n == 1:
        return [[0, 0, 1],[0, 1, 0],[1, 0, 0]]
    if n == 2:
        return [[0, 0, 2],
        [0, 1, 1],
        [0, 2, 0],
        [1, 0, 1],
        [1, 1, 0],
        [2, 0, 0]]
    if n == 3:
        return [[0, 0, 3],
        [0, 1, 2],
        [0, 2, 1],
        [0, 3, 0],
        [1, 0, 2],
        [1, 1, 1],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
        [3, 0, 0]]
    if n == 4:
        return [[0, 0, 4],
        [0, 1, 3],
        [0, 2, 2],
        [0, 3, 1],
        [0, 4, 0],
        [1, 0, 3],
        [1, 1, 2],
        [1, 2, 1],
        [1, 3, 0],
        [2, 0, 2],
        [2, 1, 1],
        [2, 2, 0],
        [3, 0, 1],
        [3, 1, 0],
        [4, 0, 0]]
    if n == 5:
        return [[0, 0, 5],
        [0, 1, 4],
        [0, 2, 3],
        [0, 3, 2],
        [0, 4, 1],
        [0, 5, 0],
        [1, 0, 4],
        [1, 1, 3],
        [1, 2, 2],
        [1, 3, 1],
        [1, 4, 0],
        [2, 0, 3],
        [2, 1, 2],
        [2, 2, 1],
        [2, 3, 0],
        [3, 0, 2],
        [3, 1, 1],
        [3, 2, 0],
        [4, 0, 1],
        [4, 1, 0],
        [5, 0, 0]]
    if n == 6:
        return [[0, 0, 6],
        [0, 1, 5],
        [0, 2, 4],
        [0, 3, 3],
        [0, 4, 2],
        [0, 5, 1],
        [0, 6, 0],
        [1, 0, 5],
        [1, 1, 4],
        [1, 2, 3],
        [1, 3, 2],
        [1, 4, 1],
        [1, 5, 0],
        [2, 0, 4],
        [2, 1, 3],
        [2, 2, 2],
        [2, 3, 1],
        [2, 4, 0],
        [3, 0, 3],
        [3, 1, 2],
        [3, 2, 1],
        [3, 3, 0],
        [4, 0, 2],
        [4, 1, 1],
        [4, 2, 0],
        [5, 0, 1],
        [5, 1, 0],
        [6, 0, 0]]
def compute_n_comb_lxlylz(n):
    #compute n!/(lx!ly!lz!), lx+ly+lz=n
    if n == 0:
        return [1.0] # 0!/(0!*0!*0!)
    if n == 1:
        #[[0, 0, 1],[0, 1, 0],[1, 0, 0]]
        return [1.0,1.0,1.0]
    if n == 2:
        #[[0, 0, 2],[0, 1, 1],[0, 2, 0],[1, 0, 1],[1, 1, 0],[2, 0, 0]]
        return [1.0, 2.0, 1.0, 2.0, 2.0, 1.0]
    if n == 3:
        #[[0, 0, 3],[0, 1, 2],[0, 2, 1],[0, 3, 0],[1, 0, 2],[1, 1, 1],[1, 2, 0],[2, 0, 1],[2, 1, 0],[3, 0, 0]]
        return [1.0,3.0,3.0,1.0,3.0,6.0,3.0,3.0,3.0,1.0]
    if n == 4:
        #[[0, 0, 4],[0, 1, 3],[0, 2, 2],[0, 3, 1],[0, 4, 0],[1, 0, 3],[1, 1, 2],[1, 2, 1],
        #[1, 3, 0],[2, 0, 2],[2, 1, 1],[2, 2, 0],[3, 0, 1],[3, 1, 0],[4, 0, 0]]
        return [1.0,4.0,6.0,4.0,1.0,4.0,12.0,12.0,4.0,6.0,12.0,6.0,4.0,4.0,1.0]
def compute_number_of_terms_lxlylz(n):
    number_of_triplets = {0: 1, 1: 3, 2: 6, 3: 10, 4: 15, 
                          5: 21, 6: 28, 7: 36, 8: 45, 9: 55, 
                          10: 66, 11: 78, 12: 91, 13: 105, 14: 120, 
                          15: 136, 16: 153, 17: 171, 18: 190, 19: 210, 20: 231}
    total = 0
    for i in range(n+1):
        total += number_of_triplets[i]
    return total

@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32),])
def find_three_non_negative_integers(n):
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
    #return valid_triplets
     
#@tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.int32),])
@tf.function
def factorial(n):
    # Create a tensor of integers from 1 to n
    # Use tf.range to create a sequence from 1 to n + 1
    numbers = tf.range(1, n + 1, dtype=tf.float32)
    # Calculate the factorial using tf.reduce_prod to multiply all elements
    result = tf.reduce_prod(numbers)
    return result

@tf.function(input_signature=[tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                             tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                             tf.TensorSpec(shape=(3,3), dtype=tf.float32),
                             tf.TensorSpec(shape=(3,), dtype=tf.int32),
                             tf.TensorSpec(shape=(None,), dtype=tf.float32),
                              ])
def generate_periodic_images_vdw(species_vectors, positions, lattice_vectors, image_range, C6):
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
    atom_positions = tf.repeat(atom_positions, 
                               tf.shape(translation_vectors)[0], axis=1)  # Shape: (n_atoms, (image_range * 2 + 1)^3, 3)
    
    
    # Generate periodic images
    # Expand the translation_vectors for proper broadcasting
    expanded_translations = tf.expand_dims(translation_vectors, axis=0)  # Shape: (1, (image_range * 2 + 1)^3, 3)
    
    # Perform the addition
    periodic_images = atom_positions + tf.tensordot(tf.cast(expanded_translations,tf.float32), 
                                                    lattice_vectors, axes=[2, 0])  # Shape: (n_atoms, (image_range * 2 + 1)^3, 3)
    
    species_vectors = tf.expand_dims(species_vectors, axis=1) # Shape: (n_atoms, 1, embedding)
    species_vectors = tf.repeat(species_vectors, tf.shape(translation_vectors)[0], axis=1) # (n_atoms, (image_range * 2 + 1)^3, embedding)

    C6_extended = tf.expand_dims(C6, axis=1)
    C6_extended = tf.repeat(C6_extended, tf.shape(translation_vectors)[0], axis=1)
    return periodic_images, species_vectors, C6_extended

@tf.function(input_signature=[tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                             tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                             tf.TensorSpec(shape=(3,3), dtype=tf.float32),
                             tf.TensorSpec(shape=(3,), dtype=tf.int32),
                             tf.TensorSpec(shape=(None,), dtype=tf.float32),
                              ])
def generate_periodic_images(species_vectors, positions, lattice_vectors, image_range, C6):
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
    atom_positions = tf.repeat(atom_positions, 
                               tf.shape(translation_vectors)[0], axis=1)  # Shape: (n_atoms, (image_range * 2 + 1)^3, 3)
    
    
    # Generate periodic images
    # Expand the translation_vectors for proper broadcasting
    expanded_translations = tf.expand_dims(translation_vectors, axis=0)  # Shape: (1, (image_range * 2 + 1)^3, 3)
    
    # Perform the addition
    periodic_images = atom_positions + tf.tensordot(tf.cast(expanded_translations,tf.float32), 
                                                    lattice_vectors, axes=[2, 0])  # Shape: (n_atoms, (image_range * 2 + 1)^3, 3)
    
    species_vectors = tf.expand_dims(species_vectors, axis=1) # Shape: (n_atoms, 1, embedding)
    species_vectors = tf.repeat(species_vectors, tf.shape(translation_vectors)[0], axis=1) # (n_atoms, (image_range * 2 + 1)^3, embedding)

    return periodic_images, species_vectors


def calculate_image_range_per_vector(cutoff, lattice_vectors):
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

@tf.function
def switch(r, rmin,rmax):
    x = (r-rmin)/(rmax-rmin)
    res  = tf.zeros(tf.shape(x))
    res = tf.where(x<=0., 1.0, -6.0*x**5+15.0*x**4-10.0*x**3+1.0)
    res = tf.where(x>1.0, 0.0, res)
    return res
@tf.function
def vdw_contribution(x):
    rij_norm = x[0]
    C6ij = x[1]
    rmin_u = x[2]
    rmax_u = x[3]
    rmin_d = x[4]
    rmax_d = x[5]


    rij_norm_inv6  = 1.0 / (tf.pow(rij_norm, 6) + 1e-12)
    #rij_norm_inv2 = rij_norm_inv * rij_norm_inv
    #rij_norm_inv6 = rij_norm_inv2 * rij_norm_inv2 * rij_norm_inv2

    energy = -(1 - switch(rij_norm, rmin_u, rmax_u)) * switch(rij_norm,rmin_d, rmax_d) * rij_norm_inv6
    energy = energy * C6ij
    energy = 0.5 * tf.reduce_sum(energy)
    return [energy]

@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32),
                             tf.TensorSpec(shape=(), dtype=tf.float32)])
def tf_fcut(r,rc):
    dim = tf.shape(r)
    pi = tf.constant(math.pi, dtype=tf.float32)
    return tf.where(r<=rc, 0.5*(1.0 + tf.cos(pi*r/rc)), tf.zeros(dim, dtype=tf.float32))
@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32),
                             tf.TensorSpec(shape=(), dtype=tf.float32)])
def _tf_fcut(r,rc):
    dim = tf.shape(r)
    #pi = tf.constant(math.pi, dtype=tf.float32)
    x = tf.tanh(1 - r / rc)
    return tf.where(r<=rc, x*x*x, tf.zeros(dim, dtype=tf.float32))

@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32),
                             tf.TensorSpec(shape=(), dtype=tf.float32)],jit_compile=True)
def tf_fcut_rbf(r,rc):
    p = tf.constant(6, dtype=tf.float32)
    x = r / rc
    x2 = x*x
    xp = x2 * x2 * x2
    xp1 = x2 * x2 * x2 * x
    xp2 = x2 * x2 * x2 * x2
    
    fc = tf.where(x <= 1.0, 1.0 - 
                  (p+1.)*(p+2.) / 2. * xp + 
                  p*(p+2.)*xp1 - 
                  p*(p+1)/2*xp2, tf.zeros_like(x))
    return fc

@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32)])
def tf_app_gaussian(x):
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
@tf.function(input_signature=[tf.TensorSpec(shape=(None,), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.float32),
                              tf.TensorSpec(shape=(None,), dtype=tf.float32),
                              tf.TensorSpec(shape=(), dtype=tf.int32),
                              ], jit_compile=True)
def bessel_function(r,rc,kn,n):
    
    pi = tf.constant(math.pi, dtype=tf.float32)
    nkn_rad = tf.range(1, n+1, dtype=tf.float32) * kn
    rn = pi / rc * tf.einsum('j,i->ij',nkn_rad, r)

    dim = tf.shape(r)
    p = 6
    fc_over_r = tf_fcut_rbf(r,rc) / (r + 1e-20)

    return tf.sqrt(2.0/rc)*tf.einsum('ij,i->ij',tf.sin(rn),fc_over_r) # nat, nneigh,nrad

def help_func(n):
    return tf.ones(n+1, tf.int32) * n
def quad_loss(y, y_pred):
    loss = tf.reduce_mean((y - y_pred)**2)
    return loss

@tf.function(input_signature=[(tf.TensorSpec(shape=(), dtype=tf.float32),
                             tf.TensorSpec(shape=(None,), dtype=tf.float32),
                              tf.TensorSpec(shape=(None,), dtype=tf.float32))])
def force_loss(x):

    nat = tf.cast(x[0], tf.int32)
    force_ref = tf.reshape(x[1][:3*nat], (nat,3))
    force_pred = tf.reshape(x[2][:3*nat], (nat,3))
    loss = tf.reduce_mean((force_ref - force_pred)**2)
    return loss
@tf.function(input_signature=[(tf.TensorSpec(shape=(), dtype=tf.float32),
                             tf.TensorSpec(shape=(None,), dtype=tf.float32),
                              tf.TensorSpec(shape=(None,), dtype=tf.float32))])
def force_mse(x):

    nat = tf.cast(x[0], tf.int32)
    force_ref = tf.reshape(x[1][:3*nat], (nat,3))
    force_pred = tf.reshape(x[2][:3*nat], (nat,3))
    fmse = tf.reduce_mean((force_ref - force_pred)**2)
    return fmse
@tf.function(input_signature=[(tf.TensorSpec(shape=(), dtype=tf.float32),
                             tf.TensorSpec(shape=(None,), dtype=tf.float32),
                              tf.TensorSpec(shape=(None,), dtype=tf.float32))])
def force_mae(x):

    nat = tf.cast(x[0], tf.int32)
    force_ref = tf.reshape(x[1][:3*nat], (nat,3))
    force_pred = tf.reshape(x[2][:3*nat], (nat,3))
    fmae = tf.reduce_mean(tf.abs(force_ref - force_pred))
    return fmae

