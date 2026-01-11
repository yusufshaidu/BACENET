import tensorflow as tf
import sys

@tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                 )
def _atom_centered_polarization(Rij,
                            shell_displacement,
                            z_charge, q_charge,
                            first_atom_idx,
                            second_atom_idx,
                            atomic_numbers,
                            central_atom_id):
    '''this is based on sin fourier series of a linear potential in the 
    P_ia{\sum_{j in Neigh(i)} qj Rij - Zj dj}
    '''
    shell_disp_shape = tf.shape(shell_displacement)
    nat = shell_disp_shape[0]
    n_shells = shell_disp_shape[1]
    Rij_norm = tf.linalg.norm(Rij, axis=-1) #npair
    shell_displacement = tf.reshape(shell_displacement, [nat,n_shells*3])
    
    central_atom_mask = (atomic_numbers == central_atom_id)
    central_atom_mask2idx = tf.cast(central_atom_mask, tf.float32) * 10000
    idx_central = tf.range(nat)[central_atom_mask]
    
    shell_displacement_i = shell_displacement[central_atom_mask]
    z_charge_i = z_charge[central_atom_mask]
    q_charge = tf.gather(q_charge, second_atom_idx)
    z_charge = tf.gather(z_charge, second_atom_idx)
    shell_displacement = tf.gather(shell_displacement, second_atom_idx)


    
    idx_central_pair = tf.where(first_atom_idx == idx_central[:,None])[:,1]
    first_idx_central = tf.gather(first_atom_idx, idx_central_pair)
    second_idx_central = tf.gather(second_atom_idx, idx_central_pair)

    Ri_central_j = tf.gather(Rij, idx_central_pair)
    Ri_central_j_norm = tf.gather(Rij_norm, idx_central_pair)
    qi_central_j = tf.gather(q_charge, idx_central_pair)
    zi_central_j = tf.gather(z_charge, idx_central_pair)
    shell_displacement = tf.gather(shell_displacement, idx_central_pair)

    min_dist = tf.math.unsorted_segment_min(Ri_central_j_norm,
                                 segment_ids=second_idx_central,
                                 num_segments=nat) * (1. - central_atom_mask2idx)

    # make sure that central atoms have zero distances. This is becuase the neighborlist exclude self-interactions
    # This produce the distance of all atoms from the central atoms
    # Next is to deteramined how many we have per atom
    min_dist *= 1.8 #add a buffer
    min_dist = tf.gather(min_dist, second_idx_central)
    
    #set central atom distance to large number
    shape = tf.shape(Ri_central_j_norm)
    Ri_central_j_norm = tf.where(min_dist < 0,
                                 tf.ones(shape[0]) * 1e8,
                                 Ri_central_j_norm)

    mask = Ri_central_j_norm <= min_dist
    mask2idx = tf.cast(mask, dtype=tf.float32)

    count = tf.math.unsorted_segment_sum(mask2idx,
                                         segment_ids=second_idx_central,
                                         num_segments=nat)
    
    count = tf.gather(count,second_idx_central)[mask]
    Ri_central_j = Ri_central_j[mask]
    qi_central_j = qi_central_j[mask]
    zi_central_j = zi_central_j[mask]
    shell_displacement = tf.reshape(shell_displacement[mask], [-1,n_shells,3])

    weights = 1.0 / (count + 1e-12)

    Pi_q = tf.reduce_sum(Ri_central_j * 
                         (qi_central_j + zi_central_j)[:,None] * 
                         weights[:,None], axis=0)

    Pi_e = -tf.reduce_sum((Ri_central_j[:,None,:] + shell_displacement) * 
                         zi_central_j[:,None,None] * 
                         weights[:,None,None], axis=0)

    # we should add the self correction in the main function
    Pi_self = -tf.reduce_sum(shell_displacement_i * z_charge_i[:,None], axis=0)

    return Pi_q, Pi_e + tf.reshape(Pi_self, [n_shells,3])

@tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32),
                ]
                 )
def _atom_centered_dV(Rij,
                    shell_displacement,
                    z_charge, q_charge,
                    first_atom_idx,
                    second_atom_idx,
                    atomic_numbers,
                    central_atom_id,
                    efield):
    '''this is based on sin fourier series of a linear potential in the
    dP_ia = {\sum_{j in Neigh(i)} qj Rij - Zj dj}
    '''
    shell_disp_shape = tf.shape(shell_displacement)
    nat = shell_disp_shape[0]
    n_shells = shell_disp_shape[1]

    Rij_norm = tf.linalg.norm(Rij, axis=-1) #npair
    central_atom_mask = (atomic_numbers == central_atom_id)
    central_atom_mask2idx = tf.cast(central_atom_mask, tf.float32) * 10000
    idx_central = tf.range(nat)[central_atom_mask]

    idx_central_pair = tf.where(first_atom_idx == idx_central[:,None])[:,1]
    first_idx_central = tf.gather(first_atom_idx, idx_central_pair)
    second_idx_central = tf.gather(second_atom_idx, idx_central_pair)
    Ri_central_j = tf.gather(Rij, idx_central_pair)

    z_charge_i = z_charge[central_atom_mask]
    z_charge_j = tf.gather(z_charge, second_atom_idx)
    zi_central_j = tf.gather(z_charge_j, idx_central_pair)

    Ri_central_j_norm = tf.gather(Rij_norm, idx_central_pair)
    shape = tf.shape(Ri_central_j_norm)
    min_dist = tf.math.unsorted_segment_min(Ri_central_j_norm,
                                 segment_ids=second_idx_central,
                                 num_segments=nat) * (1. - central_atom_mask2idx)

    # make sure that central atoms have zero distances. This is becuase the neighborlist exclude self-interactions
    # This produce the distance of all atoms from the central atoms
    # Next is to deteramined how many we have per atom
    min_dist *= 1.8 #add a buffer
    min_dist = tf.gather(min_dist, second_idx_central)

    #set central atom distance to large number
    Ri_central_j_norm = tf.where(min_dist < 0,
                                 tf.ones(shape[0]) * 1e8,
                                 Ri_central_j_norm)

    mask = Ri_central_j_norm <= min_dist
    Ri_central_j = Ri_central_j[mask]
    zi_central_j = zi_central_j[mask]

    mask2idx = tf.cast(mask, dtype=tf.float32)
    count = tf.math.unsorted_segment_sum(mask2idx,
                                         segment_ids=second_idx_central,
                                         num_segments=nat)
    count = tf.gather(count,second_idx_central)[mask]

    weights = tf.math.divide_no_nan(1.0, count)

    rij_field = tf.squeeze(tf.matmul(Ri_central_j, efield[:,None]))
    dViq = -tf.math.unsorted_segment_sum(rij_field * weights,
                                         segment_ids=second_idx_central[mask],
                                         num_segments=nat)
    
    #can be computed in the unit cell
    #dVie = tf.reduce_sum(efield[None,None,:] * weights_ik[:,:,None] * z_charge[None,:,None], axis=0)
    dVie = tf.math.unsorted_segment_sum(zi_central_j * weights,
                                         segment_ids=second_idx_central[mask],
                                         num_segments=nat)[:,None] * efield[None,:]
    # we should add the self correction in the main function
    dVie += (tf.cast(central_atom_mask, tf.float32)[:,None] *
             z_charge[:,None] * efield[None,:])
    return dViq, tf.reshape(tf.tile(dVie[:,None,:], [1,n_shells,1]), [-1,n_shells*3])

@tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(3,), dtype=tf.float32),
                ]
                 )
def _atom_centered_dV(Rij,
                    shell_displacement,
                    z_charge, q_charge,
                    first_atom_idx,
                    second_atom_idx,
                    atomic_numbers,
                    central_atom_id,
                    efield):
    shell_disp_shape = tf.shape(shell_displacement)
    nat = shell_disp_shape[0]
    n_shells = shell_disp_shape[1]

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(q_charge)
        tape.watch(shell_displacement)
        Piq, Pie = atom_centered_polarization(Rij,
                                    shell_displacement,
                                    z_charge, q_charge,
                                    first_atom_idx,
                                    second_atom_idx,
                                    atomic_numbers,
                                    central_atom_id)
        energy_field = -(tf.reduce_sum(Piq * efield) +
                         tf.reduce_sum(Pie * efield[None,:]))
    dViq = tape.gradient(energy_field, q_charge)
    dVie = tape.gradient(energy_field, shell_displacement)
    return dViq, tf.reshape(dVie, [-1,n_shells*3])
@tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(3,3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                )
def atom_centered_polarization(shell_displacement, positions,
                                z_charge, q_charge,
                                central_atom_id,
                                atomic_numbers, cell,
                                n_shells=1):

    """
    A vectorized function for polarization
    """
    n_shells = tf.shape(shell_displacement)[1]
    z_charge_shells = tf.tile(z_charge[:,None], [1,n_shells])
    nat = tf.shape(q_charge)[0]

    #z_charge = tf.cond(tf.greater(n_shells, 1), lambda: tf.ones((self._n_atoms, n_shells)) * 2.0,
    #        lambda: z_charge[:,None])
    #if tf.greater(n_shells, 1):
    #    z_charge = tf.ones((self._n_atoms, n_shells)) * 2.0 # 2.0 for each shell
    #else:
    #    z_charge = z_charge[:,None]
    #cell = self._cell
    r = tf.range(-1, 2)
    X, Y, Z = tf.meshgrid(r, r, r, indexing='ij')
    replicas = tf.stack([tf.reshape(X, [-1]),
                   tf.reshape(Y, [-1]),
                   tf.reshape(Z, [-1])], axis=1)
    #R_vector = tf.matmul(replicas, cell) # shape = 27,3
    R_vector = tf.matmul(tf.cast(replicas,tf.float32), cell) # shape = 27,3

    i_idx = (atomic_numbers == central_atom_id)
    positions_i = tf.reshape(positions[i_idx], [-1,3]) # nat_c,3
#    positions_i = tf.stop_gradient(positions_i)
    nat_c = tf.shape(positions_i)[0]

    unique_type, unique_idx, counts = tf.unique_with_counts(atomic_numbers)
    idx_cen = (unique_type == central_atom_id)

    composition = tf.cast(counts / tf.reduce_min(counts),
                                tf.float32)
    composition_cent = composition[idx_cen]
    #initialize P
    position_replicas = R_vector[None,:,:] + positions[:,None,:] # nat, 27, 3
    Rij = position_replicas[None,...] - positions_i[:,None, None,:] # nat_c,nat,27,3
    Rij_norm = tf.linalg.norm(Rij, axis=3) # nat_c, nat, 27

    #compute the minimum distance per atom for each central atom and thier replicas
    # include a buffer of less the minimum distance itself. Because, if rmin_i is the distance of atom i from the central atom,
    # atoms at r < rmin are images
    min_r = tf.reduce_min(
        tf.reshape(
            tf.transpose(Rij_norm, [1,0,2]),
                   [-1,nat_c*27]),
        axis=1) * 1.8 # [nat_c, nat]
    #get a mask of valid atoms
    mask = tf.less_equal(Rij_norm, min_r[None,:,None])

    #To determin the weights, we sum over all true values
    _count_selected = tf.reduce_sum(tf.cast(mask, tf.float32), axis=-1)  # [nat_c,nat] float
    #How many centra atoms do I share
    valid_count_selected = tf.reduce_sum(_count_selected,
                                   axis=0)[None,:] * tf.cast(_count_selected>=1,
                                                             dtype=tf.float32)
    weights_ij = tf.math.divide_no_nan(tf.ones((nat_c,nat)) , valid_count_selected)
    #weights_ij = 1.0 / (valid_count_selected + 1e-12) #(nat_c,nat)
    #tf.print(valid_count_selected[0], output_stream=sys.stderr)

    #Rij_masked = tf.where(mask[..., None], Rij, tf.zeros((nat_c,nat, 27, 3)))
    mask_int = tf.where(mask)
    weights_ij = tf.gather_nd(weights_ij[:,:,None] * tf.cast(mask, tf.float32), mask_int)
    weights_ij = tf.stop_gradient(weights_ij)

    # scale by the weights(the number of images per atom)
    Rij_masked = tf.gather_nd(Rij, mask_int)
    avg_Rij = Rij_masked * weights_ij[:,None]             # [nat_c,nat,3]
    #qz = tf.gather_nd(tf.tile((q_charge + z_charge)[None,:],
    q = tf.gather_nd(tf.tile(q_charge[None,:],
                              [nat_c,1])[:,:,None] *
                      tf.cast(mask, tf.float32), mask_int)
    #z_shell = tf.gather_nd(tf.tile(z_charge_shells[None,:,:],
    #                          [nat_c,1,1])[:,:,None,:] *
    #                  tf.cast(mask, tf.float32)[...,None], mask_int)
    shell_displacement = tf.gather_nd(tf.tile(shell_displacement[None,:,:,:],
                                              [nat_c,1,1,1])[:,:,None,:,:] *
                      tf.cast(mask, tf.float32)[...,None,None], mask_int)

    #print(qz, z_shell)
    Piq = tf.reduce_sum(q[:,None] * avg_Rij, axis=0) / composition_cent
    # normalization, useful if there are more than 1 central atom type per formular unit
    #computing shell contribution
    # We rewrite P_shell = -\sum_c \sum_{j in uc} Z_j\sum_{R} (Rcj + dj + R)
    #Rij_shell_sum = tf.reduce_sum(Rij_shell_masked, axis=2) # sum over replicas
    #avg_Rij_shell = Rij_shell_masked * weights_ij[:,None,None]

    Pie = -2.0 * tf.reduce_sum(shell_displacement * 
                               weights_ij[:,None,None], axis=0) / composition_cent # n_shells,3
    return Piq, tf.reshape(Pie, [-1,3])

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(3,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            ]
            )
def atom_centered_dV(shell_displacement,positions,
                    z_charge, q_charge,
                    central_atom_id,
                    atomic_numbers,
                    cell,efield,
                    n_shells=1):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(q_charge)
        tape.watch(shell_displacement)
        Piq, Pie = atom_centered_polarization(
                                shell_displacement,
                                positions,
                                z_charge, q_charge,
                                central_atom_id,
                                atomic_numbers,cell,
                                n_shells=n_shells)
        energy_field = -(tf.reduce_sum(Piq * efield) +
                         tf.reduce_sum(Pie * efield[None,:]))
    dViq = tape.gradient(energy_field, q_charge)
    dVie = tape.gradient(energy_field, shell_displacement)
    return dViq, tf.reshape(dVie, [-1,n_shells*3])
########### Not used
@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            ]
             )
def polarization(charges, 
                 positions, 
                 shell_disp):
    '''
    P = sum_i q_i R_i -2*sum_{in} p_in (=Zidi). 
    Zi = 2 for each shell
    '''
    N = tf.shape(positions)[0]
    Rij = positions[None,:,:] - positions[:,None,:]

    Piq = tf.reduce_sum(charges[None,:,None] * Rij, axis=(0,1)) / tf.cast(N, tf.float32)

    Pie = -2.0 * tf.reduce_sum(shell_disp, axis=0) #nshells, 3
    return Piq, Pie

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            ]
             )
def dV(positions, efield, n_shells=1):
    '''
    dE_field/dq = -sum_a(R_ia . e_field_a) 
    dE_field/dd_in = e_field * ones(N,shell,3). 
    Zi = 2 for each shell
    '''
    N = tf.shape(positions)[0]
    Rij = positions[None,:,:] - positions[:,None,:]

    dVdq = -tf.reduce_sum(Rij * efield[None,None,:], axis=(0,2)) / tf.cast(N, tf.float32)
    dVdp = 2.0 * tf.ones((N,n_shells,3)) * efield[None,None,:] #N, nshells, 3
    return dVdq, tf.reshape(dVdp, [N,n_shells*3])

