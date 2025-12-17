import tensorflow as tf
from models.coulomb_functions import (_compute_energy_exact,
                                      _compute_charges)

def solver(E_d2, z_charge, atomic_q0, 
           total_charge, b, E2, n_shells, ewald,
           tol=1e-3, max_iter=50):
    #
    z_charge_n = tf.tile(z_charge[:,None], [1,n_shells]) / tf.cast(n_shells, tf.float32)
    N = tf.shape(atomic_q0)[0]
    charges0 = atomic_q0    
    shell_disp = tf.ones((N, 3*n_shells)) * 0.01
    count0 = tf.constant(0)
    conv0 = False
    E0 = 0.0
    E_d2 = tf.reshape(E_d2, [N, n_shells, 3])
    def _body(shell_disp, charges, count, converged):
        bb = tf.identity(b)
        charges_old = tf.identity(charges)
        Vij_qq, Vij_qz, Vijn_qz, Vij_zq, Vijn_zq, dE_d_d, ddE_dd_d = \
                ewald.recip_space_term_with_shelld_exact_derivatives(shell_disp, z_charge, charges, E_d2)

        bb += 0.5 * tf.reduce_sum(
                (tf.transpose(Vij_zq * z_charge[:,None], perm=(1, 0)) + Vij_qz * z_charge[None,:]),
            axis=1
        )
        bb += 0.5 * tf.reduce_sum(
                (tf.transpose(Vijn_zq * z_charge_n[:,None,:], perm=(1, 0, 2)) + Vijn_qz * z_charge_n[None,:,:]),
            axis=(1, 2)
        )

        charges = _compute_charges(Vij_qq, bb, E2, atomic_q0, total_charge)
        dshell_disp = -tf.reshape(tf.linalg.solve(ddE_dd_d, dE_d_d[:,:,None]),
                                  [N,3*n_shells])
        converged = (tf.linalg.norm(dshell_disp) + 
                     tf.linalg.norm(charges-charges_old)) < tol
        return shell_disp + dshell_disp, charges, count + 1, converged

    def single_iter(shell_disp, charges):
        bb = tf.identity(b)
        charges_old = tf.identity(charges)
        Vij_qq, Vij_qz, Vijn_qz, Vij_zq, Vijn_zq, dE_d_d, ddE_dd_d = \
                ewald.recip_space_term_with_shelld_exact_derivatives(shell_disp, z_charge, charges, E_d2)
                
        bb += 0.5 * tf.reduce_sum(
                (tf.transpose(Vij_zq * z_charge[:,None], perm=(1, 0)) + Vij_qz * z_charge[None,:]),
            axis=1
        )
        bb += 0.5 * tf.reduce_sum(
                (tf.transpose(Vijn_zq * z_charge_n[:,None,:], perm=(1, 0, 2)) + Vijn_qz * z_charge_n[None,:,:]),
            axis=(1, 2)
        )
      
        charges = _compute_charges(Vij_qq, bb, E2, atomic_q0, total_charge)
        dshell_disp = -tf.reshape(tf.linalg.solve(ddE_dd_d, dE_d_d[:,:,None]),
                                  [N,3*n_shells])
        return shell_disp + dshell_disp, charges



    def _cond(shell_disp, charges, count, converged):
        return tf.logical_and(
            tf.logical_not(converged),
            count < max_iter
        )

    #Execute SCF
    shell_disp_final, charges, _, _ = tf.while_loop(
        _cond,
        _body,
        loop_vars=[shell_disp, charges0, count0, conv0],
        parallel_iterations	= 1,
        maximum_iterations=max_iter
    )
    #shell_disp_final, charges = single_iter(shell_disp, charges0)
    #shell_disp_final, charges = tf.zeros_like(shell_disp), charges0
    _V = ewald.recip_space_term_with_shelld_exact(shell_disp_final)
    Vij_qq, Vij_qz, Vijn_qz, Vij_zq, Vijn_zq, Vij_zz, Vijn_zz1, Vijn_zz2, Vijnn_zz = _V

    E = _compute_energy_exact(Vij_qq, Vij_qz, Vijn_qz, Vij_zq,
                           Vijn_zq, Vij_zz, Vijn_zz1,
                           Vijn_zz2, Vijnn_zz,
                           E_d2, shell_disp_final,
                           charges, z_charge)

    return charges, shell_disp_final, E

