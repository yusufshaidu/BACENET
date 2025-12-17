import tensorflow as tf

class ConjugateGradientSolver:
    def __init__(self, A_matvec, M_inv, tol=1e-6, maxiter=1000):
        self.A_matvec = A_matvec
        self.M_inv = M_inv
        self.tol = tol
        self.maxiter = maxiter

    # Diagonal preconditioner
    #def M_inv(self, r):
    #    diag_inv = 1.0 / (tf.linalg.diag_part(self.A_mat) + 1e-6)
    #    return diag_inv * r

    # Matrix–vector product
    #def A_matvec(self, x):
    #    return tf.linalg.matvec(self.A_mat, x)

    def solve(self, b, x0=None):
        b = tf.convert_to_tensor(b, dtype=tf.float32)

        if x0 is None:
            x0 = tf.zeros_like(b)
        else:
            x0 = tf.convert_to_tensor(x0, dtype=tf.float32)

        # Initial residual
        r0 = b - self.A_matvec(x0)
        z0 = self.M_inv(r0)
        p0 = tf.identity(z0)
        rsold0 = tf.reduce_sum(r0 * z0)

        # Loop condition
        def cond(i, x, r, z, p, rsold):
            converged = tf.sqrt(tf.reduce_sum(r * r)) < self.tol
            return tf.logical_and(i < self.maxiter, tf.logical_not(converged))

        # Loop body
        def body(i, x, r, z, p, rsold):
            Ap = self.A_matvec(p)
            alpha = rsold / tf.reduce_sum(p * Ap)

            x_new = x + alpha * p
            r_new = r - alpha * Ap

            z_new = self.M_inv(r_new)
            rsnew = tf.reduce_sum(r_new * z_new)
            beta = rsnew / rsold

            p_new = z_new + beta * p

            return i+1, x_new, r_new, z_new, p_new, rsnew

        # Run TF while loop
        i_final, x_final, r_final, z_final, p_final, rs_final = tf.while_loop(
            cond,
            body,
            loop_vars=[0, x0, r0, z0, p0, rsold0],
            parallel_iterations=1
        )

        return x_final
