import tensorflow as tf



@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ]
             )
def _compute_Aij( Vij, E2):
    '''construct Aij'''

    #this is removed after padding Vij with 1's at the last row and columns
    # Aij has exactly zero at N+1,N+1 elements a needed
   
    N = tf.shape(E2)[0]
    E2_padded = tf.concat([E2, [-1.0]], axis=0)  # shape [N+1]
    Aij = tf.concat([Vij, tf.ones(N,tf.float32)[:,None]], 1)
    Aij = tf.concat([Aij, tf.ones(N+1,tf.float32)[None,:]], 0)
    Aij += tf.linalg.diag(E2_padded)
    return Aij # (N+1,N+1)
@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            ]
             )
def _compute_Fia( Vij_qz, E_qd):
    '''construct Aij'''

    #this is removed after padding Vij with 1's at the last row and columns
    # Aij has exactly zero at N+1,N+1 elements a needed
   
    N = tf.shape(Vij_qz)[0]

    delta_ij = tf.eye(N)

    Fija = 0.5 * Vij_qz # N,N,3
    E_qd = E_qd[:,None,:] * delta_ij[:,:,None] # N,N,3      
    Fija += E_qd * 0.5
    Fija = tf.pad(tf.reshape(Fija,[N, N*3]), [[0,1],[0,0]])

    return tf.reshape(Fija,[N+1,3*N]) # N+1, N * 3

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            ]
             )
def _compute_Fiajb(Vij_zz,E_di):
    '''construct Aij'''

    #this is removed after padding Vij with 1's at the last row and columns
    # Aij has exactly zero at N+1,N+1 elements a needed
   
    N = tf.shape(E_di)[0]
    #Eijab = E_di delta_ij delta_ab
    Eijab = E_di[:,None,:,None] * tf.eye(N)[:,:,None,None] * tf.eye(3)[None,None,:,:]
    #Eijab = E_di[:,None,:,:] * tf.eye(N)[:,:,None,None]
    Fiajb = tf.transpose(Vij_zz + Eijab, perm=(0,2,1,3)) # N,N,3,3
    return tf.reshape(Fiajb, [3*N, 3*N]) # 3*N, 3*N

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            ]
             )
def _compute_charges_disp(Vij, Vij_qz, Vij_zq, Vij_zz, 
                         E1, E2,E_d2, E_d1, E_qd,
                         atomic_q0,total_charge):
    '''compute charges and shell displacements through the solution of linear system'''

    #collect all block matrices
    Vij_qz = Vij_qz + tf.transpose(Vij_zq, perm=(1,0,2))
    N = tf.shape(atomic_q0)[0]
    Aij = _compute_Aij(Vij, E2) #shape = (N+1,N+1)
    Fija = _compute_Fia(Vij_qz, E_qd) # shape = (N+1,3N)
    Fiajb = _compute_Fiajb(Vij_zz, E_d2) # shape = (3N,3N)
    upper_layer = tf.concat([Aij, Fija], axis=1)
    lower_layer = tf.concat([tf.transpose(Fija), Fiajb], axis=1)
    Mat = tf.concat([upper_layer, lower_layer], axis=0)
    E1_padded = tf.concat([-E1, [total_charge]], axis=0)
    b = tf.concat([E1_padded, -tf.reshape(E_d1, [-1])], axis=0)

    charges_disp = tf.squeeze(tf.linalg.solve(Mat, b[:,None]))

    '''
    lin_op_A = tf.linalg.LinearOperatorFullMatrix(
        Aij,
        is_self_adjoint=True,
        is_positive_definite=True,  # Optional: set to True if you know it is
        is_non_singular=True        # Optional: set to True if you know it is
    )
    outs = tf.linalg.experimental.conjugate_gradient(
        lin_op_A,
        E1,
        preconditioner=None,
        x=atomic_q0,
        tol=1e-05,
        max_iter=500,
        name='conjugate_gradient'
        )
    #outs[0]= max_iter, outs[2]=residual,outs[3]=basis vectors, outs[4]=preconditioner 
    charges = outs[1]
    '''
    charges = charges_disp[:N]
    shell_disp = tf.reshape(charges_disp[N+1:], [N,3])
    return charges, shell_disp

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            ]
             )
def _compute_shell_disp_qqdd2(Vij, Vij_qz, Vij_zq, Vij_zz, 
                             Vij_qz2, Vij_zq2, Vij_qq2, Vij_qq3, E_d1,
                       E_d2, E_qd, atomic_q0,charges, field_kernel_e, field_kernel_qe):
    '''comput charges through the solution of linear system'''
    dq = charges - atomic_q0
    nat = tf.shape(dq)[0]
    
    charge_ij = charges[:,None] * charges[None,:]
    A_ia = 0.5 * tf.reduce_sum((tf.transpose(Vij_qz,perm=(1,0,2)) + Vij_zq) * charges[None,:,None], axis=1) #N,3
    A_ia += 0.5 * tf.reduce_sum((tf.transpose(Vij_qq2,perm=(1,0,2)) - Vij_qq2) * charge_ij[:,:,None], axis=1) #N,3

    A_iab = tf.reduce_sum((tf.transpose(Vij_qz2,perm=(1,0,2,3)) + Vij_zq2) * charges[None,:,None,None], axis=1) #N,3,3
    A_iab += 0.5 * tf.reduce_sum((tf.transpose(Vij_qq3,perm=(1,0,2,3)) + Vij_qq3) * charge_ij[:,:,None,None], axis=1) #N,3,3
    A_ijab = Vij_zz + A_iab * tf.eye(nat)[:,:,None,None]
    A_ijab += E_d2[:,None,:,None] * tf.eye(nat)[:,:,None,None] * tf.eye(3)[None,None,:,:]
    #A_ijab -= 2.0 * 0.5 * (tf.transpose(Vij_qz2,perm=(1,0,3,2)) + Vij_zq2) * charges[None,:,None,None] #N,N,3,3
    A_ijab -= (tf.transpose(Vij_qz2,perm=(1,0,3,2)) + Vij_zq2) * charges[None,:,None,None] #N,N,3,3
    A_ijab -= (tf.transpose(Vij_zq2,perm=(1,0,3,2)) + Vij_qz2) * charges[:,None,None,None] #N,N,3,3
    A_ijab -= 2. * Vij_qq3 * charge_ij[:,:,None,None]

    A_iajb = tf.reshape(tf.transpose(A_ijab, [0,2,1,3]), [nat*3, nat*3])
    A_ia += E_d1 +  0.5 * E_qd * dq[:,None] + field_kernel_e + charges[:,None] * field_kernel_qe
    A_ia = tf.reshape(A_ia, [-1])
    shell_disp = tf.reshape(tf.linalg.solve(A_iajb, -A_ia[:,None]), [nat,3])
    return shell_disp

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            ]
             )
def _compute_shell_disp_qqdd1(V11, V21, V12, V22, V3, E_d1,
                       E_d2, E_qd, atomic_q0,charges, field_kernel_e):
    '''comput charges through the solution of linear system'''
    dq = charges - atomic_q0
    nat = tf.shape(dq)[0]
    A_ia = 0.5 * tf.reduce_sum((tf.transpose(V11,perm=(1,0,2)) + V21) * charges[None,:,None], axis=1) #N,3
    A_iab = tf.reduce_sum((tf.transpose(V12, perm=(1,0,2,3)) + V22) * charges[None,:,None,None], axis=1) #N,3,3

    A_ijab = (V3 + 
              A_iab * tf.eye(nat)[:,:,None,None] + 
              E_d2[:,None,:,None] * tf.eye(nat)[:,:,None,None] * tf.eye(3)[None,None,:,:])
    
    A_iajb = tf.reshape(tf.transpose(A_ijab, [0,2,1,3]), [nat*3, nat*3])
    A_ia += E_d1 +  0.5 * E_qd * dq[:,None] + field_kernel_e
    A_ia = tf.reshape(A_ia, [-1])
    shell_disp = tf.reshape(tf.linalg.solve(A_iajb, -A_ia[:,None]), [nat,3])
    return shell_disp

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            ]
             )
def _compute_charges(Vij, E1, E2, atomic_q0,total_charge):
    '''comput charges through the solution of linear system'''

    #this is removed after padding Vij with 1's at the last row and columns
    # Aij has exactly zero at N+1,N+1 elements a needed
   
    N = tf.shape(E2)[0]
    E2_padded = tf.concat([E2, [-1.0]], axis=0)  # shape [N+1]
    Aij = tf.concat([Vij, tf.ones(N,tf.float32)[:,None]], 1)
    Aij = tf.concat([Aij, tf.ones(N+1,tf.float32)[None,:]], 0)
    Aij += tf.linalg.diag(E2_padded)

    #Aij = 0.5 * (Aij + tf.transpose(Aij))

    #total charge should be read from structures
    E1_padded = tf.concat([-E1, [total_charge]], axis=0)
    #since we are fitting dq rather than q, total charges must be replaced with zero
    #E1_padded = tf.concat([-E1, [0.0]], axis=0)
    #atomic_q0 = tf.pad(atomic_q0, [[0,1]], constant_values=0.0)
    #atomic_q0 = tf.concat([atomic_q0, [0.0]], axis=0)
    
    charges = tf.linalg.solve(Aij, E1_padded[:,None])
    '''
    lin_op_A = tf.linalg.LinearOperatorFullMatrix(
        Aij,
        is_self_adjoint=True,
        is_positive_definite=True,  # Optional: set to True if you know it is
        is_non_singular=True        # Optional: set to True if you know it is
    )
    outs = tf.linalg.experimental.conjugate_gradient(
        lin_op_A,
        E1,
        preconditioner=None,
        x=atomic_q0,
        tol=1e-05,
        max_iter=500,
        name='conjugate_gradient'
        )
    #outs[0]= max_iter, outs[2]=residual,outs[3]=basis vectors, outs[4]=preconditioner 
    charges = outs[1]
    '''
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
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            ]
             )
def _compute_coulumb_energy_pqeq_qd(charges, atomic_q0, 
                                E1, E2, shell_disp, Vij_qq, Vij_qz, Vij_zq, Vij_zz):
    '''compute the coulumb energy
    Energy = \sum_i E_1i q_i + E_2i q^2/2 + \sum_i\sum_j Vij qiqj / 2 + \
            \sum_i\sum_j\sum_a Vija_qz shell_disp_ia qizj/2 + \sum_i\sum_j \sum_ab Vijab_zz shell_disp_ia * shell_disp_jb zizj
    '''
    #q = charges
    shell_disp_outer = shell_disp[:,None,:,None] * shell_disp[None,:,None,:] # N,N,3,3

    dq = charges - atomic_q0
    dq2 = dq * dq
    q_outer = charges[:,None] * charges[None,:]

    E = E1 * dq
    qi_dj = charges[:,None,None] * shell_disp[None,:,:] #N,N,3
    qj_di = charges[None,:,None] * shell_disp[:,None,:] #N,N,3
    E += 0.5 * (E2 * dq2 + tf.reduce_sum(
        Vij_qq * q_outer + 
        tf.reduce_sum(Vij_qz * qi_dj + Vij_zq * qj_di, axis=2) + 
        tf.reduce_sum(Vij_zz * shell_disp_outer, axis=(2,3)),
        axis=-1
        )
                         )
    return tf.reduce_sum(E)

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

@tf.function(jit_compile=False,
             input_signature=[
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            ])
def run_scf(Vij, Vij_qz, Vij_zq, Vij_zz, Vij_qz2, Vij_zq2,
    E_d1, E_d2, E_qd, E2,
    atomic_q0, total_charge,
    field_kernel_qe, field_kernel_e,b,
    tol=1e-6, max_iter=50):
    """
    Performs SCF loop for charges and shell displacements.
    """

    # Initial conditions
    nat = tf.shape(E_d1)[0]
    charges0 = atomic_q0
    shell_disp0 = tf.zeros((nat,3))
    count0 = tf.constant(0)
    conv0 = tf.constant(False)
    def scf_cond(charges_old, shell_disp_old, count, converged):
        return tf.logical_and(
            tf.logical_not(converged),
            count < max_iter
        )

    def scf_body(charges_old, shell_disp_old, count, converged):

        # --- Start SCF iteration ---
        bb = tf.identity(b)

        # 1. Compute new shell displacement d(q)
        shell_disp = _compute_shell_disp_qqdd1(
            Vij_qz, Vij_zq, Vij_qz2, Vij_zq2, Vij_zz,
            E_d1, E_d2 + field_kernel_qe, E_qd,
            atomic_q0, charges_old, field_kernel_e
        )

        # 2. Quadratic displacement tensor (outer product)
        shell_d2 = shell_disp[:, :, None] * shell_disp[:, None, :]

        # 3. Update b with Z–q and Z–Z couplings
        bb += 0.5 * tf.reduce_sum(
            (tf.transpose(Vij_zq, perm=(1, 0, 2)) + Vij_qz) * shell_disp[None, :, :],
            axis=(1, 2)
        )
        bb += 0.5 * tf.reduce_sum(
            (tf.transpose(Vij_zq2, perm=(1, 0, 2, 3)) + Vij_qz2) * shell_d2[None, ...],
            axis=(1, 2, 3)
        )

        # 4. Add q–d terms
        bb += 0.5 * tf.reduce_sum(E_qd * shell_disp, axis=1)

        # 5. Solve for new charges
        charges = _compute_charges(Vij, b, E2, atomic_q0, total_charge)

        # --- Convergence check ---
        res = tf.linalg.norm(charges - charges_old) + tf.linalg.norm(shell_disp - shell_disp_old)

        converged = res < tol

        return (
            charges,
            shell_disp,
            count + 1,
            converged
        )

    # Execute SCF
    charges_final, shell_final, _, _ = tf.while_loop(
        scf_cond,
        scf_body,
        loop_vars=[charges0, shell_disp0, count0, conv0],
        maximum_iterations=max_iter
    )

    return charges_final, shell_final

