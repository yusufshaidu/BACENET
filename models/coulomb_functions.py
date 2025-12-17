import tensorflow as tf
from models.cg import ConjugateGradientSolver

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
    Aij = tf.concat([Vij, tf.ones(N,tf.float32)[:,None]], 1)
    Aij = tf.concat([Aij, tf.ones(N+1,tf.float32)[None,:]], 0)
    Aij += tf.linalg.diag(E2_padded)
    return Aij # (N+1,N+1)

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            ]
             )
def _compute_Fia(Vij_qz, field_kernel_qe):
    '''construct Aij'''

    #this is removed after padding Vij with 1's at the last row and columns
    # Aij has exactly zero at N+1,N+1 elements a needed
    N = tf.shape(Vij_qz)[0]
    n_3 = tf.shape(Vij_qz)[2]
    Fija = 0.5 * Vij_qz # N,N,n_3
    Fija += 0.5 * field_kernel_qe[:,None,:] * tf.eye(N)[:,:,None]
    Fija = tf.pad(tf.reshape(Fija,[N, N*n_3]), [[0,1],[0,0]])
    return tf.reshape(Fija,[N+1,n_3*N]) # N+1, N * n_3

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            ]
             )
def _compute_Fiajb(Vij_zz,E_di):
    '''construct Aij'''

    #this is removed after padding Vij with 1's at the last row and columns
    # Aij has exactly zero at N+1,N+1 elements a needed
   
    N = tf.shape(E_di)[0]
    n_3 = tf.shape(Vij_zz)[2] #n * 3
    #Eijab = E_di delta_ij delta_ab
    Eijab = E_di[:,None,:,None] * tf.eye(N)[:,:,None,None] * tf.eye(n_3)[None,None,:,:]
    #Eijab = E_di[:,None,:,:] * tf.eye(N)[:,:,None,None]
    Fiajb = tf.transpose(Vij_zz + Eijab, perm=(0,2,1,3)) # N,N,3,3
    return tf.reshape(Fiajb, [n_3*N, n_3*N]) # 3*N, 3*N

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            ]
             )
def _compute_charges_disp(Vij, Vij_qz, Vij_zq, Vij_zz, 
                         E1, E2, E_d2, field_kernel_e, 
                        field_kernel_qe,
                         atomic_q0,total_charge):
    '''compute charges and shell displacements through the solution of linear system'''

    #collect all block matrices
    N = tf.shape(E1)[0]
    n = tf.shape(Vij_qz)[2]
    
    #E_d1 = tf.tile(E_d1[:,:,None], [1,1,n])
    #E_d2 = tf.tile(E_d2[:,:,None], [1,1,n])

    #E_d1 = tf.reshape(E_d1, [N,3*n])
    #E_d2 = tf.reshape(E_d2, [N,3*n])

    Vij_qz = tf.reshape(Vij_qz, [N,N,n*3])
    Vij_zq = tf.reshape(Vij_zq, [N,N,n*3])
    Vij_zz = tf.reshape(tf.transpose(Vij_zz, perm=(0,1,2,4,3,5)), [N,N,n*3,n*3])
    Vij_qz = Vij_qz + tf.transpose(Vij_zq, perm=(1,0,2))
    Aij = _compute_Aij(Vij, E2) #shape = (N+1,N+1)
    Fija = _compute_Fia(Vij_qz,field_kernel_qe) # shape = (N+1,3nN)
    Fiajb = _compute_Fiajb(Vij_zz, E_d2) # shape = (3nN,3nN)
    upper_layer = tf.concat([Aij, Fija], axis=1)
    lower_layer = tf.concat([tf.transpose(Fija), Fiajb], axis=1)
    def A_matvec(v):
        Ax11 = tf.linalg.matvec(Aij, v[:N+1]) 
        Ax12 = tf.linalg.matvec(Fija, v[N+1:])
        #Ax_upper = tf.concat([Ax11, Ax12], axis=0)
        Ax21 = tf.linalg.matvec(tf.transpose(Fija), v[:N+1]) 
        Ax22 = tf.linalg.matvec(Fiajb, v[N+1:])
    
        Ax = tf.concat([Ax11 + Ax12, Ax21 + Ax22], axis=0)
        return Ax

    #preconditional:
    def M_inv(v):
        M = 1.0 / (tf.linalg.diag_part(Aij) + 1e-6)
        M = tf.concat([M, 1.0/(tf.linalg.diag_part(Fiajb) + 1e-6)], axis=0)
        return M * v

    Mat = tf.concat([upper_layer, lower_layer], axis=0)
    E1_padded = tf.concat([-E1, [total_charge]], axis=0)
    b = tf.concat([E1_padded, -tf.reshape(field_kernel_e, [-1])], axis=0)

    charges_disp = tf.squeeze(tf.linalg.solve(Mat, b[:,None]))
    #charges_disp = ConjugateGradientSolver(A_matvec, M_inv, tol=1e-3, maxiter=50).solve(b)

    charges = charges_disp[:N]
    shell_disp = tf.reshape(charges_disp[N+1:], [N,3*n])
    return charges, shell_disp

@tf.function(jit_compile=False,
            input_signature=[
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,3,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            ]
             )
def _compute_shell_disp_qqdd1(V11, V21, V12, V22, V3,
                       E_d2, atomic_q0,charges, field_kernel_e):
    '''comput charges through the solution of linear system'''
    dq = charges - atomic_q0
    nat = tf.shape(dq)[0]
    A_ia = 0.5 * tf.reduce_sum((tf.transpose(V11,perm=(1,0,2)) + V21) * charges[None,:,None], axis=1) #N,3
    A_iab = tf.reduce_sum((tf.transpose(V12, perm=(1,0,2,3)) + V22) * charges[None,:,None,None], axis=1) #N,3,3

    A_ijab = (V3 + 
              A_iab * tf.eye(nat)[:,:,None,None] + 
              E_d2[:,None,:,None] * tf.eye(nat)[:,:,None,None] * tf.eye(3)[None,None,:,:])
    
    A_iajb = tf.reshape(tf.transpose(A_ijab, [0,2,1,3]), [nat*3, nat*3])
    A_ia += field_kernel_e
    A_ia = tf.reshape(A_ia, [-1])
    shell_disp = tf.reshape(tf.linalg.solve(A_iajb, -A_ia[:,None]), [nat,3])
    return shell_disp

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
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None), dtype=tf.float32),
            tf.TensorSpec(shape=(None,None,None,None), dtype=tf.float32),
            ]
             )
def _compute_coulumb_energy_pqeq_qd(charges, atomic_q0, 
                                E1, E2, shell_disp, Vij_qq, Vij_qz, Vij_zq, Vij_zz):
    '''compute the coulumb energy
    Energy = \sum_i E_1i q_i + E_2i q^2/2 + \sum_i\sum_j Vij qiqj / 2 + \
            \sum_i\sum_j\sum_a Vija_qz shell_disp_ia qizj/2 + \sum_i\sum_j \sum_ab Vijab_zz shell_disp_ia * shell_disp_jb zizj
    '''
    #q = charges
    shell_disp_outer = shell_disp[:,None,:,None] * shell_disp[None,:,None,:] # N,N,?,?

    dq = charges - atomic_q0
    dq2 = dq * dq
    q_outer = charges[:,None] * charges[None,:]

    E = E1 * dq
    qi_dj = charges[:,None,None] * shell_disp[None,:,:] #N,N,?
    qj_di = charges[None,:,None] * shell_disp[:,None,:] #N,N,?
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
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(None,3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            ])
def run_scf(Vij, Vij_qz, Vij_zq, 
            Vij_zz, Vij_qz2, Vij_zq2,
            E_d2, E2,
            atomic_q0, total_charge,
            field_kernel_e,b,
            tol=1e-6, max_iter=50):
    """
    Performs SCF loop for charges and shell displacements.
    """

    # Initial conditions
    nat = tf.shape(E_d2)[0]
    charges0 = atomic_q0
    shell_disp0 = tf.zeros((nat,3))
    count0 = tf.constant(0)
    conv0 = tf.constant(False)
    charges = _compute_charges(Vij, _b, E2, total_charge)
    def scf_cond(charges_old, shell_disp_old, count, converged):
        return tf.logical_and(
            tf.logical_not(converged),
            count < max_iter
        )

    def scf_body(charges_old, shell_disp_old, count, converged):

        # --- Start SCF iteration ---
        bb = b

        shell_disp = _compute_shell_disp_qqdd1(
            Vij_qz, Vij_zq, Vij_qz2, Vij_zq2, Vij_zz,
            E_d2, atomic_q0, charges_old, field_kernel_e
        )

        shell_d2 = shell_disp[:, :, None] * shell_disp[:, None, :]

        bb += 0.5 * tf.reduce_sum(
            (tf.transpose(Vij_zq, perm=(1, 0, 2)) + Vij_qz) * shell_disp[None, :, :],
            axis=(1, 2)
        )
        bb += 0.5 * tf.reduce_sum(
            (tf.transpose(Vij_zq2, perm=(1, 0, 2, 3)) + Vij_qz2) * shell_d2[None, ...],
            axis=(1, 2, 3)
        )

        charges = _compute_charges(Vij, b, E2, total_charge)

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
