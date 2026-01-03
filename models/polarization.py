import tensorflow as tf

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
def atom_centered_polarization(Rij,
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
def atom_centered_dV(Rij,
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
