import tensorflow as tf

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ]
             )
def _compute_Aij(Vij, E2):
    '''construct Aij'''

    #this is removed after padding Vij with 1's at the last row and columns
    # Aij has exactly zero at N+1,N+1 elements a needed
   
    N = tf.shape(E2)[0]
    E2_padded = tf.concat([E2, [-1.0]], axis=0)  # shape [N+1]
    Aij = tf.concat([Vij, tf.ones(N, tf.float32)[:,None]], 1)
    Aij = tf.concat([Aij, tf.ones(N+1,tf.float32)[None,:]], 0)
    Aij += tf.linalg.diag(E2_padded)
    return Aij # (N+1,N+1)

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
            ]
             )
def _compute_Fia(Vij_qz):
    '''construct Aij'''

    #this is removed after padding Vij with 1's at the last row and columns
    # Aij has exactly zero at N+1,N+1 elements a needed
    N = tf.shape(Vij_qz)[0]
    n_3 = tf.shape(Vij_qz)[2]

    Fija = 0.5 * Vij_qz # N,N,n_3
    #Fija = tf.pad(tf.reshape(Fija,[N, N*3]), [[0,1],[0,0]])
    Fija = tf.reshape(Fija, [N, N*n_3])
    zero_row = tf.zeros((1, N*n_3), dtype=Fija.dtype)
    Fija = tf.concat([Fija, zero_row], axis=0)
    return tf.reshape(Fija,[N+1,N*n_3]) # N+1, N * n_3

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            ]
             )
def _compute_Fiajb(Vij_zz, E_di):
    '''construct Aij'''
    #this is removed after padding Vij with 1's at the last row and columns
    E_di = tf.reshape(E_di, [-1])

    N_3 = tf.shape(E_di)[0]
    Fiajb = tf.reshape(tf.transpose(Vij_zz, perm=(0,2,1,3)),
                       [N_3,N_3]) # 3nN,3nN
    #Eijab = E_di delta_iajb
    Fiajb += E_di[:,None] * tf.eye(N_3)
    return tf.reshape(Fiajb, [N_3,N_3]) # 3*N, 3*N
@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            ]
             )
def _compute_charges_disp(Vij, Vija, Vijab, 
                         E1, E2, E_d2, field_kernel_e, 
                         atomic_q0,total_charge):
    '''compute charges and shell displacements through the solution of linear system'''
    #collect all block matrices
    N = tf.shape(E1)[0]
    Aij = _compute_Aij(Vij, E2) #shape = (N+1,N+1)
    Fija = _compute_Fia(Vija) # shape = (N+1,3N)
    Fiajb = _compute_Fiajb(Vijab, E_d2) # shape = (3N,3N)
    upper_layer = tf.concat([Aij, Fija], axis=1)
    lower_layer = tf.concat([tf.transpose(Fija), Fiajb], axis=1)

    Mat = tf.concat([upper_layer, lower_layer], axis=0)
    #E1_padded = tf.concat([-E1, [total_charge]], axis=0)
    b = tf.concat([-E1, [total_charge],-tf.reshape(field_kernel_e, [-1])], axis=0)
    charges_disp = tf.squeeze(tf.linalg.solve(Mat, b[:,None]))
    charges = charges_disp[:N]
    shell_disp = tf.reshape(charges_disp[N+1:], [N,3])
    return charges, shell_disp


@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            ]
             )
def _compute_charges_disp_n(Vij, Vija, Vijab, 
                         E1, E2, E_d2, field_kernel_e, 
                         atomic_q0,total_charge):
    '''compute charges and shell displacements through the solution of linear system'''
    #collect all block matrices
    N = tf.shape(E1)[0]
    Aij = _compute_Aij(Vij, E2) #shape = (N+1,N+1)
    Fija = _compute_Fia(Vija) # shape = (N+1,3nN)
    Fiajb = _compute_Fiajb(Vijab, E_d2) # shape = (3nN,3nN)
    upper_layer = tf.concat([Aij, Fija], axis=1)
    lower_layer = tf.concat([tf.transpose(Fija), Fiajb], axis=1)

    Mat = tf.concat([upper_layer, lower_layer], axis=0)
    #E1_padded = tf.concat([-E1, [total_charge]], axis=0)
    b = tf.concat([-E1, [total_charge],-tf.reshape(field_kernel_e, [-1])], axis=0)
    charges_disp = tf.squeeze(tf.linalg.solve(Mat, b[:,None]))
    charges = charges_disp[:N]
    shell_disp = tf.reshape(charges_disp[N+1:], [N,-1])
    return charges, shell_disp

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            ]
             )
def _compute_charges(Vij, E1, E2, total_charge):
    '''comput charges through the solution of linear system'''

    #this is removed after padding Vij with 1's at the last row and columns
    # Aij has exactly zero at N+1,N+1 elements a needed
   
    N = tf.shape(E2)[0]
    E2_padded = tf.concat([E2, [-1.0]], axis=0)  # shape [N+1]
    Aij = tf.concat([Vij, tf.ones(N,tf.float32)[:,None]], 1)
    Aij = tf.concat([Aij, tf.ones(N+1,tf.float32)[None,:]], 0)
    Aij += tf.linalg.diag(E2_padded)

    #total charge should be read from structures
    E1_padded = tf.concat([-E1, [total_charge]], axis=0)
    #L = tf.linalg.cholesky(Aij)
    #charges = tf.linalg.cholesky_solve(L, E1_padded[:,None])
    charges = tf.linalg.solve(Aij, E1_padded[:,None])
    return tf.reshape(charges, [-1])[:-1]
#"""
@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            ]
             )
def _compute_coulumb_energy(charges, atomic_q0, E1, E2, Vij):
    '''compute the coulumb energy
    Energy = \sum_i E_1i q_i + E_2i q^2/2 + \sum_i\sum_j Vij qiqj / 2
    '''
    q = charges
    dq = q - atomic_q0
    dq2 = dq * dq
    q_outer = q[:,None] * q[None,:]
    E = E1 * dq + 0.5 * (E2 * dq2 + tf.reduce_sum(Vij * q_outer, axis=-1))
    return tf.reduce_sum(E)
@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            ]
             )
def _compute_coulumb_energy_pqeq_qd(charges, atomic_q0, 
                                E1, E2, shell_disp, Vij, Vija, Vijab):
    '''compute the coulumb energy
    Energy = \sum_i E_1i q_i + E_2i q^2/2 + \sum_i\sum_j Vij qiqj / 2 + \
            \sum_i\sum_j\sum_a Vija_qz pj qi/2 + \sum_i\sum_j \sum_ab Vijab_zz shell_disp_ia * shell_disp_jb zizj
    '''
    #q = charges
    shell_disp = tf.reshape(shell_disp, [-1])
    N_3 = tf.shape(shell_disp)[0]
    Vijab_transpose = tf.reshape(tf.transpose(Vijab, [0,2,1,3]), 
                                 [N_3, N_3])
    Vija_transpose = tf.reshape(Vija, [-1, N_3])

    shell_disp_outer = shell_disp[:,None] * shell_disp[None,:] # N_3,N_3
    q_outer = charges[:,None] * charges[None,:]
    qi_dj = charges[:,None] * shell_disp[None,:] #N,N_3,?


    dq = charges - atomic_q0
    dq2 = dq * dq
    
    E = tf.reduce_sum(E1 * dq + 0.5 * E2 * dq2)
    E += 0.5 * tf.reduce_sum(Vij * q_outer)
    E += 0.5 * tf.reduce_sum(Vija_transpose * qi_dj) 
    E += 0.5 * tf.reduce_sum(Vijab_transpose * shell_disp_outer)
    return E
@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None,None), dtype=tf.float32),
            ]
             )
def _compute_coulumb_energy_pqeq_qd_n(charges, atomic_q0, 
                                E1, E2, shell_disp, Vij, Vija, Vijab):
    '''compute the coulumb energy
    Energy = \sum_i E_1i q_i + E_2i q^2/2 + \sum_i\sum_j Vij qiqj / 2 + \
            \sum_i\sum_j\sum_a Vija_qz pj qi/2 + \sum_i\sum_j \sum_ab Vijab_zz shell_disp_ia * shell_disp_jb zizj
    '''
    #q = charges
    shell_disp = tf.reshape(shell_disp, [-1])
    N_3 = tf.shape(shell_disp)[0]
    Vijab_transpose = tf.reshape(tf.transpose(Vijab, [0,2,1,3]), 
                                 [N_3, N_3])
    Vija_transpose = tf.reshape(Vija, [-1, N_3])

    shell_disp_outer = shell_disp[:,None] * shell_disp[None,:] # N_3,N_3
    q_outer = charges[:,None] * charges[None,:]
    qi_dj = charges[:,None] * shell_disp[None,:] #N,N_3,?


    dq = charges - atomic_q0
    dq2 = dq * dq
    
    E = tf.reduce_sum(E1 * dq + 0.5 * E2 * dq2)
    E += 0.5 * tf.reduce_sum(Vij * q_outer)
    E += 0.5 * tf.reduce_sum(Vija_transpose * qi_dj) 
    E += 0.5 * tf.reduce_sum(Vijab_transpose * shell_disp_outer)
    return E

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            ]
             )
def _compute_coulumb_energy_pqeq(charges, atomic_q0, nuclei_charge, 
                                E1, E2, Vij_qq, Vij_qz, Vij_zq, Vij_zz):
    '''compute the coulumb energy
    Energy = \sum_i E_1i q_i + E_2i q^2/2 + \sum_i\sum_j Vij qiqj / 2 + \
            \sum_i\sum_j Vij_qz qizj/2 + \sum_i\sum_j Vij_zz zizj
    '''
    #q = charges
    dq = charges - atomic_q0
    dq2 = dq * dq
    q_outer = charges[:,None] * charges[None,:]
    qz_outer = charges[:,None] * nuclei_charge[None,:]
    zq_outer = charges[None, :] * nuclei_charge[:,None]
    zz_outer = nuclei_charge[:,None] * nuclei_charge[None,:]
    #if self.learn_oxidation_states:
    #    E = E1 * tf.abs(dq)
    #else:
    E = E1 * dq

    E += 0.5 * (E2 * dq2 + tf.reduce_sum(
        Vij_qq * q_outer + Vij_qz * qz_outer + Vij_zq * zq_outer + Vij_zz * zz_outer, axis=-1
        )
                         )
    return tf.reduce_sum(E)

