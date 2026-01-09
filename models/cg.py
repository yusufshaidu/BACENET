import tensorflow as tf

@tf.function
def cg_solve(A, b, max_iter=50, tol=1e-6):
    x = tf.zeros_like(b)
    r = b
    p = r

    def cond(i, x, r, p):
        r_norm = tf.linalg.norm(r)
        return tf.logical_and(i < max_iter, r_norm > tol)

    def body(i, x, r, p):
        Ap = A(p)
        alpha = tf.reduce_sum(r*r) / (tf.reduce_sum(p*Ap) + 1e-12)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = tf.reduce_sum(r_new*r_new) / (tf.reduce_sum(r*r) + 1e-12)
        p = r_new + beta * p
        return i+1, x, r_new, p

  #  max_it, x, _, _ = tf.while_loop(cond, body, [0, x, r, p], back_prop=False)
    max_it, x, _, _ = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(cond, body, [0, x, r, p]))
    return max_it, x

@tf.function
def cg_precond_solve(A, b, M=None, max_iter=50, tol=1e-6):
    """
    A: callable, A(x)
    M: callable or None, preconditioner M(x) ~ A^{-1} x
    """
    x = tf.zeros_like(b)
    r = b
    z = M(r)
    p = z
    gamma = tf.reduce_sum(r * z)

    def cond(i, x, r, z, p, gamma):
        r_norm = tf.linalg.norm(r)
        return tf.logical_and(i < max_iter, r_norm > tol)

    def body(i, x, r, z, p, gamma):
        Ap = A(p)

        alpha = gamma / (tf.reduce_sum(p * Ap) + 1e-12)

        x = x + alpha * p
        r_new = r - alpha * Ap

        z_new = M(r_new)
        #z_new = M * r_new

        gamma_new = tf.reduce_sum(r_new * z_new)
        beta = gamma_new / (gamma + 1e-12)
        p = z_new + beta * p

        return i + 1, x, r_new, z_new, p, gamma_new

    max_it, x, _, _, _, _ = tf.nest.map_structure(tf.stop_gradient, tf.while_loop(
        cond,
        body,
        [0, x, r, z, p, gamma],
        parallel_iterations=1
    ))

    return max_it, x


