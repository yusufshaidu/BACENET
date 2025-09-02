import tensorflow as tf
import numpy as np
import scipy.constants as constants
import math
from functions.solver import newton_solve_spd
from functions.solver_general import newton_solve

#pi = tf.constant(math.pi)
pi = 3.141592653589793

constant_e = 1.602176634e-19
constant_eps = 8.8541878128e-12
CONV_FACT = 1e10 * constant_e / (4 * pi * constant_eps)

class ewald:
    '''
    A python on class to implement ewald sum with gaussian charge distribution
    '''
    def __init__(self, positions, cell, n_atoms, gaussian_width,
                 accuracy, gmax,pbc=False,efield=None,
                 gaussian_width_scaling=1.0):
        '''
           positions: atomic positions in Angstrom
           cell: cell dimension
           gaussian_width: gaussian width for each atom
           gmax: maximum G for the the reciprocal summation
           rmax: cutoff distance for the short range term
        '''
        self._cell = tf.convert_to_tensor(cell, dtype=tf.float32)
        self._n_atoms = tf.cast(n_atoms, tf.int32)
        self._positions = tf.convert_to_tensor(positions, dtype=tf.float32)
        self._accuracy = accuracy
        #self._shell_displacement = tf.convert_to_tensor(shell_displacement, dtype=tf.float32)
        
        #self._pair_displacement = pair_displacement
        #self._first_atom_idx = first_atom_idx

        self._gaussian_width = tf.convert_to_tensor(gaussian_width * gaussian_width_scaling, 
                                                    dtype=tf.float32) # this could be learnable, maybe!
        self._sqrt_pi = tf.sqrt(pi)
        

        self.volume = tf.abs(tf.linalg.det(self._cell))

        #structure properties
        self.cell_length = tf.linalg.norm(self._cell,axis=1)
        self.pbc = pbc if pbc else False

        if self.pbc:
            self.reciprocal_cell = 2 * pi * tf.transpose(tf.linalg.inv(self._cell))
            gamma_max = 1/(tf.sqrt(2.0)*tf.reduce_min(self._gaussian_width))
            self._gmax = gmax if gmax else 2 * gamma_max * tf.sqrt(-tf.math.log(accuracy))
            self._gmax *= 1.0001
        if efield is not None:
            self.efield = tf.cast(tf.squeeze(efield), dtype=tf.float32)
            self.reciprocal_cell = 2 * pi * tf.transpose(tf.linalg.inv(self._cell))
        else:
            self.efield = [0.,0.,0.]
        g_vecs, g_norm = self.compute_valid_gvec_norm()
        self.g_vecs = g_vecs
        self.g_norm = g_norm


    def real_space_term(self):
        '''
        calculates the self interaction contribution to the electrostatic energy
        Args:
            All lengths are in angstrom
            gaussian_width: gaussian width, array of atomic gussian width
            coords : atomic coordinates
        Return:
            real space contribution to energy
        '''

        #convert to angstrom * eV so that q1q2/r is in eV when r is in angstrom and
        # q1, q2 are in electronic charge unit
        #1e10 comes from angstrom
        #CONV_FACT = 1e10 * e_charge / (4 * np.pi * epsilon_0)

        rij = self._positions[:,None,:] - self._positions[None,:,:]
        gamma_ij = tf.sqrt(self._gaussian_width[:,None]**2 + self._gaussian_width**2)

        #compute the norm of all distances
        rij_norm = tf.linalg.norm(rij + 1e-12, axis=-1)
        #trick to set the i==j element of the matrix to zero
        rij_norm_inv = 1 / (rij_norm+1e-12)

        #erf_term = np.where(rij_norm>0, erf(gamma_all*rij_norm)/rij_norm, 0)
        erf_term = tf.math.erf(rij_norm/gamma_ij/tf.sqrt(2.0)) * rij_norm_inv
        Vij = tf.identity(erf_term)

        
        E_diag = 1. / self._sqrt_pi * 1.0/(self._gaussian_width)
        Vij = tf.linalg.set_diag(Vij, E_diag) * CONV_FACT

        ''' 
        if compute_forces:
            #rij_tmp = np.where(rij_norm>0, 1/rij_norm**2,0)

            rij_tmp = rij_norm_inv * rij_norm_inv
            F_erf_part = erf_term * rij_tmp
            F_gaussian_part = -2.0/np.sqrt(np.pi) * gamma_all * np.exp(-gamma_all**2*rij_norm**2) * rij_tmp
            F_norm = F_erf_part + F_gaussian_part
            F_norm = np.tile(F_norm[:,:,np.newaxis], 3)
            force = rij * F_norm
            return [V, CONV_FACT*force, rij, gamma_all]
        '''
        return Vij

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                ]
                 )
    def estimate_gmax(self,b_norm, sigma_min):
        gmax = 1.0
        result = 1e2
        # convergence are based on forces: ~ 1 / volume * sum_G sum_j G_alpha/G^2 exp(-G^2*(alpha_i^2 + alpha_j^2)/4) sin(G.rij)
        #energy goes as 1/G^2 but note that the forces norm goes as 1/G
        c = lambda gmax, result: tf.greater(result, self._accuracy)
        b = lambda gmax, result: (gmax + 1.0, 
                                  1.0 / self.volume * tf.exp(-gmax**2*b_norm**2*sigma_min**2/2.) / (gmax**2*b_norm**2))
        return tf.while_loop(c, b, [gmax, result])[0]
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                 )
    def generate_g_vectors(self, gmax, nmax1, nmax2, nmax3):
        g_vecs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        g_norm = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        index = 0
        for i in tf.range(-nmax1, nmax1 + 1):
            for j in tf.range(-nmax2, nmax2 + 1):
                for k in tf.range(-nmax3, nmax3 + 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    n = tf.cast([i, j, k], tf.float32)
                    g = tf.linalg.matvec(self.reciprocal_cell, n)  # [3]
                    _g_norm = tf.linalg.norm(g)
                    if _g_norm <= gmax:
                        g_vecs = g_vecs.write(index, g)
                        g_norm = g_norm.write(index, _g_norm)
                        index += 1
        return [g_vecs.stack(), g_norm.stack()]  # [K, 3], [K]
    @tf.function
    def accumulate_over_g(self, rij_flat, sigma_flat, g_vecs, g_sq, chunk_size):
        N2 = tf.shape(rij_flat)[0]  # N*N
        K = tf.shape(g_vecs)[0]
        #g_sq = g_norm * g_norm  # [K]
        Vij_flat = tf.zeros([N2], dtype=rij_flat.dtype)

        def cond(i, acc):
            return i < K

        def body(i, acc):
            end = tf.minimum(i + chunk_size, K)
            g_chunk = g_vecs[i:end]           # [M,3]
            g_sq_chunk = g_sq[i:end]          # [M]

            dot = tf.matmul(rij_flat, tf.transpose(g_chunk))  # [N2, M]
            cos_term = tf.cos(dot)  # [N2, M]

            exp_arg = -0.25 * tf.expand_dims(sigma_flat, 1) * tf.expand_dims(g_sq_chunk, 0)  # [N2, M]
            exp_term = tf.exp(exp_arg)
            inv_g_sq = 1.0 / (tf.expand_dims(g_sq_chunk, 0) + 1e-12)  # [1, M]

            contribution = exp_term * inv_g_sq * cos_term  # [N2, M]
            acc += tf.reduce_sum(contribution, axis=1)  # accumulate over k in chunk
            return end, acc

        _, Vij_flat = tf.while_loop(
            cond,
            body,
            loop_vars=[0, Vij_flat],
            shape_invariants=[tf.TensorShape([]), tf.TensorShape([None])]
        )
        return Vij_flat  # later reshape to [N, N]

    @tf.function
    def recip_space_term(self): 
        '''
        calculates the interaction contribution to the electrostatic energy
        '''


        #compute the norm of all distances
        # the force on k due to atoms j is
        # rk-ri
        rij = self._positions[:,None,:] - self._positions[None,:,:]
        rij = tf.reshape(rij, [-1,3])
        sigma_ij2 = self._gaussian_width[:,None]**2 + self._gaussian_width**2
        sigma_ij2 = tf.reshape(sigma_ij2, [-1])
        

        '''
        #gmax = self._gmax

        # Build integer index grid for k-vectors
        # Estimate index ranges for each dimension
        #b1_norm,b2_norm,b3_norm = tf.linalg.norm(self.reciprocal_cell, axis=-1)
        b_norm = tf.linalg.norm(self.reciprocal_cell, axis=-1)
        b1_norm = b_norm[0]
        b2_norm = b_norm[1]
        b3_norm = b_norm[2]
        sigma_min = tf.reduce_min(self._gaussian_width)
        gmax1 = self.estimate_gmax(b1_norm, sigma_min)
        gmax1 *= b1_norm 
        gmax2 = self.estimate_gmax(b2_norm, sigma_min)
        gmax2 *= b2_norm 
        gmax3 = self.estimate_gmax(b3_norm, sigma_min)
        gmax3 *= b3_norm 
        gmax = tf.reduce_max([gmax1,gmax2,gmax3])
        #gmax = tf.reduce_max([gmax1,gmax2])


        nmax1 = tf.cast(tf.math.floor(gmax / b1_norm), tf.int32)
        nmax2 = tf.cast(tf.math.floor(gmax / b2_norm), tf.int32)
        nmax3 = tf.cast(tf.math.floor(gmax / b3_norm), tf.int32)
        n1 = tf.range(-nmax1, nmax1 + 1, dtype=tf.int32)
        n2 = tf.range(-nmax2, nmax2 + 1, dtype=tf.int32)
        n3 = tf.range(-nmax3, nmax3 + 1, dtype=tf.int32)

        # Meshgrid of indices
        n1g, n2g, n3g = tf.meshgrid(n1, n2, n3, indexing='ij')
        replicas = tf.stack([tf.reshape(n1g, [-1]),
                            tf.reshape(n2g, [-1]),
                            tf.reshape(n3g, [-1])], axis=1)  # [M,3]
        replicas = tf.cast(replicas, tf.float32)
        g_all = tf.matmul(replicas, self.reciprocal_cell)  # [M,3] = n · recip_cell (each row n * recip_cell)
        nonzero = tf.reduce_any(tf.not_equal(replicas, 0), axis=1)
        # Mask: exclude (0,0,0) and enforce |k|<=kmax
        g_vecs = tf.boolean_mask(g_all, nonzero)
        g_norm = tf.linalg.norm(g_vecs, axis=1)
        
        #mask = tf.logical_and(g_norm <= gmax, nonzero)
        mask = g_norm <= gmax
        g_vecs = tf.boolean_mask(g_vecs, mask)  # [K,3]
        g_norm = tf.boolean_mask(g_norm, mask) #[K,]
        #g_vecs, g_norm = self.generate_g_vectors(gmax, nmax1, nmax2, nmax3)
        # keep g where first nonzero component is positive,
        # tie-breakers: if g[0]==0, require g[1]>0; if g[0]==g[1]==0, require g[2] >= 0
        g0, g1, g2 = tf.unstack(g_vecs, axis=1)
        mask = (
            (g0 > 0) |
            ((g0 == 0) & (g1 > 0)) |
            ((g0 == 0) & (g1 == 0) & (g2 >= 0))
        )
        g_vecs = tf.boolean_mask(g_vecs, mask)      # [K_plus, 3]
        g_norm = tf.boolean_mask(g_norm, mask)  # [K_plus]
        '''
        g_sq = self.g_norm * self.g_norm  # [K]

        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        #g2_gamma2 = tf.einsum('ij,l->ijl', sigma_ij2/4.0, g_sq) # [N*N,K]
        #chunk_size = 128
        #Vij = self.accumulate_over_g(rij, sigma_ij2, g_vecs, g_sq, chunk_size)
        '''
        Vij = tf.zeros(self._n_atoms * self._n_atoms, dtype=tf.float32)
        #for k in tf.range(tf.shape(g_sq)[0]):
        for k in tf.range(200):
            g =  g_sq[k]
            g_v = g_vecs[k,:]
            g2_gamma2 = sigma_ij2 / 4.0 * g # [N*N,K]
            exp_ij = tf.exp(-g2_gamma2)
            exp_ij_by_g_sq = exp_ij * 1.0 / (g + 1e-12)

            # The cosine term: shape [N*N,K]
            cos_term = tf.cos(tf.einsum('ij, j->i', rij, g_v))
            #rij_dot_g = tf.squeeze(tf.matmul(rij, g_v[:,None]))
            #cos_term = tf.cos(rij_dot_g)
            #cos_term = tf.cos(tf.reduce_sum(rij[:,:,None,:] * g_vecs[None,None,:,:], axis=-1))

            # Build energy contribution for each g as exp(-g^2 * gamma_ij**2/4)/g_sq[k] * cos_term[k]
            # Compute exp outer product and divide by k^2
            Vij += exp_ij_by_g_sq * cos_term  # [N*N]
        ''' 

        g2_gamma2 = 0.25 * tf.einsum('i,j->ij',sigma_ij2, g_sq) # [N*N,K]
        exp_ij = tf.exp(-g2_gamma2)
        #exp_ij_by_g_sq = exp_ij * 1.0 / (g_sq[None,:] + 1e-12)
        g_sq_inv = 1.0 / (g_sq + 1e-12)

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        rij_dot_g = tf.matmul(rij, tf.transpose(self.g_vecs))
        cos_term = tf.cos(rij_dot_g)
        #cos_term = tf.cos(tf.reduce_sum(rij[:,:,None,:] * g_vecs[None,None,:,:], axis=-1))

        # Build energy contribution for each g as exp(-g^2 * gamma_ij**2/4)/g_sq[k] * cos_term[k]
        # Compute exp outer product and divide by k^2
        Vij = 2.0 * tf.einsum('ij,ij,j->i',exp_ij, cos_term, g_sq_inv)  # [N,N]
        
        # Multiply by prefactor (charges, 4π/V, Coulomb constant e²/4πε0 in eV·Å units) so that E = 1/2 \sum_ij Vij
        Vij *= CONV_FACT * (4.* pi / self.volume)
        
        return tf.reshape(Vij, [self._n_atoms, self._n_atoms]) 
    @tf.function
    def compute_valid_gvec_norm(self):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        b_norm = tf.linalg.norm(self.reciprocal_cell, axis=-1)
        b1_norm = b_norm[0]
        b2_norm = b_norm[1]
        b3_norm = b_norm[2]
        sigma_min = tf.reduce_min(self._gaussian_width)
        gmax1 = self.estimate_gmax(b1_norm, sigma_min)
        gmax1 *= b1_norm 
        gmax2 = self.estimate_gmax(b2_norm, sigma_min)
        gmax2 *= b2_norm 
        gmax3 = self.estimate_gmax(b3_norm, sigma_min)
        gmax3 *= b3_norm 
        gmax = tf.reduce_max([gmax1,gmax2,gmax3])
        #gmax = tf.reduce_max([gmax1,gmax2])


        nmax1 = tf.cast(tf.math.floor(gmax / b1_norm), tf.int32)
        nmax2 = tf.cast(tf.math.floor(gmax / b2_norm), tf.int32)
        nmax3 = tf.cast(tf.math.floor(gmax / b3_norm), tf.int32)
        #'''
        n1 = tf.range(-nmax1, nmax1 + 1, dtype=tf.int32)
        n2 = tf.range(-nmax2, nmax2 + 1, dtype=tf.int32)
        n3 = tf.range(-nmax3, nmax3 + 1, dtype=tf.int32)

        # Meshgrid of indices
        n1g, n2g, n3g = tf.meshgrid(n1, n2, n3, indexing='ij')
        replicas = tf.stack([tf.reshape(n1g, [-1]),
                            tf.reshape(n2g, [-1]),
                            tf.reshape(n3g, [-1])], axis=1)  # [M,3]
        replicas = tf.cast(replicas, tf.float32)
        g_all = tf.matmul(replicas, self.reciprocal_cell)  # [M,3] = n · recip_cell (each row n * recip_cell)
        nonzero = tf.reduce_any(tf.not_equal(replicas, 0), axis=1)
        # Mask: exclude (0,0,0) and enforce |k|<=kmax
        g_vecs = tf.boolean_mask(g_all, nonzero)
        g_norm = tf.linalg.norm(g_vecs, axis=1)
        
        #mask = tf.logical_and(g_norm <= gmax, nonzero)
        mask = g_norm <= gmax
        g_vecs = tf.boolean_mask(g_vecs, mask)  # [K,3]
        g_norm = tf.boolean_mask(g_norm, mask) #[K,]
        #'''
        #g_vecs, g_norm = self.generate_g_vectors(gmax, nmax1, nmax2, nmax3)
        # keep g where first nonzero component is positive,
        # tie-breakers: if g[0]==0, require g[1]>0; if g[0]==g[1]==0, require g[2] >= 0
        g0, g1, g2 = tf.unstack(g_vecs, axis=1)
        mask = (
            (g0 > 0) |
            ((g0 == 0) & (g1 > 0)) |
            ((g0 == 0) & (g1 == 0) & (g2 >= 0))
        )
        g_vecs = tf.boolean_mask(g_vecs, mask)      # [K_plus, 3]
        g_norm = tf.boolean_mask(g_norm, mask)  # [K_plus]
        return g_vecs, g_norm

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld(self,shell_displacement):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        #save the position of the shell
        shell_displacement = tf.cast(shell_displacement, tf.float32)
        positions_plus_d = self._positions + shell_displacement
        #compute the norm of all distances
        # the force on k due to atoms j is

        # rk-ri for core-core, shell-core and shell-shell

        
        sigma_ij2 = self._gaussian_width[:,None]**2 + self._gaussian_width**2
        sigma_ij2 = tf.reshape(sigma_ij2, [-1])
        
        g_sq = self.g_norm * self.g_norm  # [K]

        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        #g2_gamma2 = tf.einsum('ij,l->ijl', sigma_ij2/4.0, g_sq) # [N*N,K]
        #chunk_size = 128
        #Vij = self.accumulate_over_g(rij, sigma_ij2, g_vecs, g_sq, chunk_size)
        '''
        Vij = tf.zeros(self._n_atoms * self._n_atoms, dtype=tf.float32)
        #for k in tf.range(tf.shape(g_sq)[0]):
        for k in tf.range(200):
            g =  g_sq[k]
            g_v = g_vecs[k,:]
            g2_gamma2 = sigma_ij2 / 4.0 * g # [N*N,K]
            exp_ij = tf.exp(-g2_gamma2)
            exp_ij_by_g_sq = exp_ij * 1.0 / (g + 1e-12)

            # The cosine term: shape [N*N,K]
            cos_term = tf.cos(tf.einsum('ij, j->i', rij, g_v))
            #rij_dot_g = tf.squeeze(tf.matmul(rij, g_v[:,None]))
            #cos_term = tf.cos(rij_dot_g)
            #cos_term = tf.cos(tf.reduce_sum(rij[:,:,None,:] * g_vecs[None,None,:,:], axis=-1))

            # Build energy contribution for each g as exp(-g^2 * gamma_ij**2/4)/g_sq[k] * cos_term[k]
            # Compute exp outer product and divide by k^2
            Vij += exp_ij_by_g_sq * cos_term  # [N*N]
        ''' 

        g2_gamma2 = 0.25 * tf.einsum('i,j->ij',sigma_ij2, g_sq) # [N*N,K]
        exp_ij = tf.exp(-g2_gamma2)
        #exp_ij_by_g_sq = exp_ij * 1.0 / (g_sq[None,:] + 1e-12)
        g_sq_inv = 1.0 / (g_sq + 1e-12)

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        cos_term = tf.cos(rij_dot_g)
        Vij_qq = 2.0 * exp_ij * cos_term * g_sq_inv[None,:]  # [N*N]


        rij = tf.reshape(positions_plus_d[:,None,:] - self._positions[None,:,:], [-1,3])
        rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        cos_term = tf.cos(rij_dot_g)
        _Vij_dj =  2.0 * exp_ij * cos_term * g_sq_inv[None,:]
        Vij_qz = 2.0 * (Vij_qq - _Vij_dj)  # [N*N]

        rij = tf.reshape(positions_plus_d[:,None,:] - positions_plus_d[None,:,:], [-1,3])
        rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        cos_term = tf.cos(rij_dot_g)
        Vij_zz = 2.0 * (Vij_qq / 2.0  - _Vij_dj + exp_ij * cos_term * g_sq_inv[None,:])  # [N*N]

        #cos_term = tf.cos(tf.reduce_sum(rij[:,:,None,:] * g_vecs[None,None,:,:], axis=-1))

        # Build energy contribution for each g as exp(-g^2 * gamma_ij**2/4)/g_sq[k] * cos_term[k]
        # Compute exp outer product and divide by k^2
        #Vij_qq = 2.0 * tf.einsum('ij,ij,j->i',exp_ij, cos_term, g_sq_inv)  # [N*N]
        #Vij_qz = 2.0 * tf.einsum('ij,ij,j->i',exp_ij, cos_term - cos_plus_dj_term, g_sq_inv)  # [N*N]
        #Vij_zz = 2.0 * tf.einsum('ij,ij,j->i',exp_ij, cos_term - 2 * cos_plus_dj_term + cos_plus_didj_term, g_sq_inv)  # [N*N]
        #divide Vij_qq by 2 because of the factor of 2 multiplied above

        # Multiply by prefactor (charges, 4pi/V, Coulomb constant e^2/4piepsilon_0 in eV-Å units) so that E = 1/2 \sum_ij Vij
        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        Vij_qq = tf.reshape(tf.reduce_sum(Vij_qq, axis=1) * conversion_fact, [self._n_atoms, self._n_atoms])
        Vij_qz = tf.reshape(tf.reduce_sum(Vij_qz, axis=1) * conversion_fact, [self._n_atoms, self._n_atoms])
        Vij_zz = tf.reshape(tf.reduce_sum(Vij_zz, axis=1) * conversion_fact, [self._n_atoms, self._n_atoms])
        
        return [Vij_qq, Vij_qz, Vij_zz]
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_opt_shell_func(self, z_charge, q_charge, Ei, shell_displacement):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        #save the position of the shell
        shell_displacement = tf.reshape(shell_displacement, [self._n_atoms,3])

        positions_plus_d = self._positions + shell_displacement
        #compute the norm of all distances
        # the force on k due to atoms j is
        # rk-ri for core-core, shell-core and shell-shell
        sigma_ij2 = self._gaussian_width[:,None]**2 + self._gaussian_width**2
        sigma_ij2 = tf.reshape(sigma_ij2, [-1])
        

        g_sq = self.g_norm * self.g_norm  # [K]

        g2_gamma2 = 0.25 * tf.einsum('i,j->ij',sigma_ij2, g_sq) # [N*N,K]
        exp_ij = tf.exp(-g2_gamma2)
        #exp_ij_by_g_sq = exp_ij * 1.0 / (g_sq[None,:] + 1e-12)
        g_sq_inv_g_alp = self.g_vecs / (g_sq[:,None] + 1e-12) # [K, 3]

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        rij = tf.reshape(self._positions[:,None,:] - positions_plus_d[None,:,:], [-1,3])
        rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        sin_term = tf.sin(rij_dot_g)
        sin_term_exp = exp_ij * sin_term
        #Vij_qz  =  [N^2,K] x [K,3]
        Vij_qz =  2.0 * tf.matmul(sin_term_exp, g_sq_inv_g_alp) #[N*N,3]

        rij = tf.reshape(positions_plus_d[:,None,:] - positions_plus_d[None,:,:], [-1,3])
        rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        sin_term = tf.sin(rij_dot_g)
        sin_term_exp = exp_ij * sin_term
        #Vij_zz  =  [N^2,K] x [K,3]
        Vij_zz = -2.0 * tf.matmul(sin_term_exp,  g_sq_inv_g_alp)  # [N*N,3]
        zq = tf.expand_dims(z_charge[:,None] * q_charge[None,:], axis=-1)
        zz = tf.expand_dims(z_charge[:,None] * z_charge[None,:], axis=-1)
        #terms = [N,N,3] x [N,N]
        Vi_qz = tf.reduce_sum(tf.reshape(Vij_qz, [self._n_atoms, self._n_atoms, 3]) * zq, axis=1) #[N,3]
        Vi_zz = tf.reduce_sum(tf.reshape(Vij_zz, [self._n_atoms, self._n_atoms, 3]) * zz, axis=1) #[N,3]
        #Vi_zz = tf.reduce_sum(tf.reshape(Vij_zz, [self._n_atoms, self._n_atoms, 3]) * zq[:,:,None], axis=1) #[N,3]


        # compute jacobian
        '''
        g_sq_inv_g_alp_beta = tf.reshape(g_sq_inv_g_alp[:,:,None] * self.g_vecs[:,None,:], [-1,9]) #[K,3,3]
        g2_gamma2 = 0.5 * (self._gaussian_width**2)[:,None] * g_sq[None,:] # [N,K]
        exp_ij = tf.exp(-g2_gamma2) # [N,K]
        ri_dot_g = tf.matmul(shell_displacement, g_vecs_transpose) # [N,K]
        charges2 = z_charge * (z_charge + q_charge) 
        cos_term_exp = charges2[:,None] * tf.cos(ri_dot_g) * exp_ij #[N,K]
        '''


        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        #J = tf.reshape(tf.linalg.diag(Ei), [-1,9]) 
        J = Ei
        #J += 2.0 * conversion_fact * tf.matmul(cos_term_exp, g_sq_inv_g_alp_beta) # [N,9]
        #J = tf.reshape(J, [self._n_atoms,3,3])
        Vi_qz *= conversion_fact
        Vi_zz *= conversion_fact
        #define the optimization function
        F_di = Ei * shell_displacement - Vi_qz - Vi_zz
        return tf.reshape(F_di, [self._n_atoms,3]), J
        #return tf.reshape(F_di, [self._n_atoms,3,1]), J
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_opt_shell_funct_linear(self, z_charge, q_charge, Ei):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        #compute the norm of all distances
        # the force on k due to atoms j is
        # rk-ri for core-core, shell-core and shell-shell
        sigma_ij2 = self._gaussian_width[:,None]**2 + self._gaussian_width**2
        sigma_ij2 = tf.reshape(sigma_ij2, [-1])
        

        g_sq = self.g_norm * self.g_norm  # [K]

        g2_gamma2 = 0.25 * tf.einsum('i,j->ij',sigma_ij2, g_sq) # [N*N,K]
        exp_ij = tf.exp(-g2_gamma2)
        #exp_ij_by_g_sq = exp_ij * 1.0 / (g_sq[None,:] + 1e-12)
        g_sq_inv_g_alp = self.g_vecs / (g_sq[:,None] + 1e-12) # [K, 3]

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        sin_term = tf.sin(rij_dot_g)
        cos_term = tf.cos(rij_dot_g)
        
        #sin_term_exp = exp_ij * sin_term
        #Vij_qz  =  [N^2,K] x [K,3]
        zq = tf.expand_dims(z_charge[:,None] * q_charge[None,:], axis=-1)
        zz = tf.expand_dims(z_charge[:,None] * z_charge[None,:], axis=-1)
        Fij_qz =  2.0 * tf.matmul(exp_ij * sin_term, g_sq_inv_g_alp) #[N*N,3]
        Fia_qz_0 = -tf.reduce_sum(tf.reshape(Fij_qz, [self._n_atoms, self._n_atoms, 3]) * zq, axis=1) #[N,3]

        g_sq_inv_g_alp_beta = tf.reshape(g_sq_inv_g_alp[:,:,None] * self.g_vecs[:,None,:], [-1,9]) #[K,3*3]
        #cos_term = tf.sqrt(1.0 - sin_term * sin_term + 1e-3)
        Fij_qz =  2.0 * tf.matmul(exp_ij * cos_term, g_sq_inv_g_alp_beta) #[N*N,9]
        Fiab_qz_1 = tf.reduce_sum(tf.reshape(Fij_qz, [self._n_atoms, self._n_atoms, 3, 3]) * tf.expand_dims(zq, axis=-1), axis=1) #[N,3,3]

        Fijab_zz_2 = tf.reshape(Fij_qz, [self._n_atoms, self._n_atoms, 3, 3]) * tf.expand_dims(zz, axis=-1) #[N,N,3,3]
        n_i, n_j, n_a, n_b = self._n_atoms, self._n_atoms, 3, 3

        delta_ij = tf.eye(n_i)[:, :, None, None]    # (i, j, 1, 1)
        #delta_ab = tf.eye(n_a)[None, None, :, :]    # (1, 1, a, b)

        term1 = Fiab_qz_1[None, ...] * delta_ij
        term2 = tf.linalg.diag(tf.reshape(Ei, [-1]))

        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        Fij = tf.reshape(Fijab_zz_2 + term1, [self._n_atoms*3, self._n_atoms*3]) * conversion_fact - term2
        #Fij = tf.reshape(term1, [self._n_atoms*3, self._n_atoms*3]) - term2
        
        #J = tf.reshape(tf.linalg.diag(Ei), [-1,9]) 
        return Fij, tf.reshape(Fia_qz_0 * conversion_fact, [self._n_atoms*3])
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def compute_shell_disp_linear(self, z_charge, q_charge, Ei):
        Fij, b = self.recip_space_term_with_opt_shell_funct_linear(z_charge, q_charge, Ei)
        x = tf.linalg.solve(Fij, -b[:,None])
        return tf.reshape(x, [-1,3])
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                ]
                 )
    def shell_optimization_newton(self, x, z_charge, q_charge, Ei,
                                   max_iter=1, tol=1e-6):
        
        #F is callable
        #x = tf.reshape(x, [-1])

        # because of the complexity of the computing jacobian, we have neglect off digonal which
        # implies that we are neglecting the interaction of the shell with other nuclei and shell of other atoms
        # The jacobian now has a dimension of [3N, 3] instead of [3N,3N]. The Newton updates becomes
        # xnew = xold - \sum_{alpha} J_{i\alpha\beta} F_{i\beta} 
        def cond(i, x, dx):
            return tf.logical_and(
                tf.less(i, max_iter),
                tf.greater(tf.norm(dx), tol)
                )

        def body(i, x, dx):
            Fx, J = self.recip_space_term_with_opt_shell_func(z_charge, q_charge, Ei, x)
            dx = -Fx / (Ei + 1e-20)
            x_new = x + dx
            return i + 1, x_new, dx

        i = tf.constant(0, dtype=tf.int32)
        dx = tf.ones_like(x)

        _, x_final, _ = tf.while_loop(cond, body, [i, x, dx])

        """
        for i in tf.range(max_iter):
            #with tf.GradientTape() as tape:
            #    tape.watch(x)
            Fx, J = self.recip_space_term_with_opt_shell_func(z_charge, q_charge, 
                                                               Ei, x)
            #dx = tf.linalg.solve(Jij, -Fx[:,None])
            #dx = tf.reshape(dx, [-1])
            #dx = -tf.matmul(tf.linalg.inv(J), Fx)
            dx = -Fx / (Ei + 1e-20)
            #dx = -tf.matmul(J, Fx)
            #x_new = x + tf.reshape(dx, [-1])
            x_new = x + dx
            '''
            N = self._n_atoms * 3

            tf.debugging.assert_shapes([
             (Jij, ('N', 'N')),
             (Fx, ('N',)),
             (dx, ('N',))
            ])
            #x = dx
            #x -= tf.reshape(dx, [-1])
            if tf.norm(dx) < tol:
            #    return x_new
            '''
            x = x_new
        """
        return x_final

    def sawtooth_PE(self):
        #k = n1 b1 + n2 b2 + n3 b3. We will set n1 = n2 = n3
        g = tf.reduce_sum(self.reciprocal_cell, axis=0)
        positions = self._positions
        cos_gr = tf.math.cos(positions * g[None,:])

        kernel = tf.reduce_sum(-cos_gr * self.efield[None,:] * positions , axis=1)
        return kernel
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                 )
    def sawtooth_electric_field(self, r, N=200):
        '''this is based on sin fourier series of a linear potential in the 
           [-L,L] and f(x)=f(x+2L). I have replace L with 2 pi / G
           E_alp(r) = -4 eps_alp / pi \sum_k=0^N {(-1)^k / (2k+1) * cos[(2k+1)/4 G_alp * r_alp]}
        '''
        G = 2.0 * pi / self.cell_length
        counts = tf.range(N+1, dtype=tf.float32)
        coeffs = 2 * counts + 1
        g_dot_r = r * G[None,:]
        #shape = (Nat,3,N)
        terms = tf.math.cos(coeffs[None,None,:] * g_dot_r[:,:,None]) 
        terms *= ((-1.0) ** (counts)[None,None,:] / coeffs[None,None,:])
        return -4.0 * self.efield[None,:] / pi  * tf.reduce_sum(terms, axis=2)
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                 )
    def sawtooth_linear_fourier(self, r, N=200):
        '''this is based on sin fourier series of a linear potential in the 
           [0,L] and f(x)=f(x+L). I have replace L with 2 pi / G
           E_alp(r) = 2 L/ pi**2 \sum_k=0^N {(-1)^k / (2k+1)**2 * sin[(2k+1) G_alp * r_alp]}
        '''
        G = 2.0 * pi / self.cell_length
        counts = tf.range(N+1, dtype=tf.float32)
        coeffs = 2 * counts + 1
        g_dot_r = r * G[None,:]
        #shape = (Nat,3,N)
        terms = tf.math.sin(coeffs[None,None,:] * g_dot_r[:,:,None])
        
        terms *= ((-1.0) ** (counts)[None,None,:] / coeffs[None,None,:]**2)
        inv_G = 1.0 / (G + 1e-12)
        return 4.0 / pi * inv_G[None,:]  * tf.reduce_sum(terms, axis=2) # shape (Nats,3)
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                 )
    def sawtooth_potential_fourier(self, r, N=200):
        '''this is based on sin fourier series of a linear potential in the 
           [-L,L] and f(x)=f(x+2L). I have replace L with 2 pi / G
           E_alp(r) = -4 eps_alp / pi \sum_k=0^N {(-1)^k / (2k+1) * cos[(2k+1)/4 G_alp * r_alp]}
        '''
        G = 2.0 * pi / self.cell_length
        counts = tf.range(N+1, dtype=tf.float32)
        coeffs = 2 * counts + 1
        g_dot_r = r * G[None,:]
        #shape = (Nat,3,N)
        terms = tf.math.sin(coeffs[None,None,:] * g_dot_r[:,:,None])

        terms *= ((-1.0) ** (counts)[None,None,:] / coeffs[None,None,:]**2)
        efield_by_G = self.efield / (G + 1e-12)
        args = 0.25 * coeffs[None,None,:] * coeffs[None,None,:] * G[None,:,None]**2 * self._gaussian_width[:,None,None]**2
        exp = tf.exp(-args)
        return 2.0 / pi * tf.reduce_sum(efield_by_G[None,:]  * tf.reduce_sum(terms * args, axis=2), axis=1) # shape (Nats,)
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def sawtooth_PE_pqeq_fourier(self, shell_displacement):
        #terms_j = -0.5*\sum_i \in Neigh(j) (R_j - R_i) . E(R_i)
        #g_vect = n1 b1 + n2 b2 + n3 b3. We will set n1 = n2 = n3
        #g = tf.reduce_sum(self.reciprocal_cell, axis=0)
        #g = 2.0 * pi / self.cell_length

        positions = self._positions
        #positions_plus_di = positions + self._shell_displacement
        
        #self._pair_displacement = pair_displacement
        #self._first_atom_idx = first_atom_idx
        #extended_position = tf.gather(positions, self._first_atom_idx)
        #extended_position_plus_di = tf.gather(positions_plus_di, self._first_atom_idx)


        #cos_gr = tf.math.cos(positions * g[None,:])
        #efield_1 = self.sawtooth_electric_field(extended_position, N=200) # efield at R with neighnors
        #efield_1 = self.sawtooth_electric_field(positions, N=200) # efield at R with neighnors
        #efield_2 = self.sawtooth_electric_field(positions_plus_di, N=200) # efield at R with neighnors
        #efield_2 = self.sawtooth_electric_field(positions, N=200) # efield at R without neighbor
        #V_at_Ri = self.sawtooth_potential_fourier(positions, N=200)
        #V_at_Ri_plus_di = self.sawtooth_potential_fourier(positions_plus_di, N=200)
        #efield_2 = self.sawtooth_electric_field(extended_position_plus_di, N=200) # efield at R+d
        #kernel_q = tf.identity(V_at_Ri)
        #kernel_e = -V_at_Ri_plus_di
        linear_pot_i = self.sawtooth_linear_fourier(positions)
        linear_pot_di = self.sawtooth_linear_fourier(shell_displacement)

        #kernel_q = 0.5 * tf.math.unsorted_segment_sum(data=tf.reduce_sum(efield_1 * self._pair_displacement,axis=1),
        #                                                      segment_ids=self._first_atom_idx, 
        #                                          num_segments=self._n_atoms) 
        #kernel_e = 0.5 * tf.reduce_sum(self._shell_displacement[:,None,:] * efield_2[None,:,:], axis=(1,2))

        kernel_q = -tf.reduce_sum(self.efield[None,:] * linear_pot_i, axis=1)
        #kernel_q = -tf.reduce_sum(self.efield[None,:] * positions, axis=1)
        kernel_e = tf.reduce_sum(self.efield[None,:] * linear_pot_di, axis=1) 
        #kernel_e = tf.reduce_sum(self.efield[None,:] * self._shell_displacement, axis=1) 
        return kernel_q, kernel_e
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def sawtooth_PE_pqeq(self, shell_displacement):
        #k = n1 b1 + n2 b2 + n3 b3. We will set n1 = n2 = n3
        #g = tf.reduce_sum(self.reciprocal_cell, axis=0)
        #E = -e[\sum_i [(zi+qi)V(Ri) - zi V(Ri+di)]]
        #g = 2.0 * pi / self.cell_length
        positions = self._positions
        #cos_gr = tf.math.cos(positions * g[None,:])

        #efield = self.sawtooth_electric_field(positions, N=200)
        V_i = self.sawtooth_potential_fourier(positions , N=250)
        V_di = self.sawtooth_potential_fourier(positions + shell_displacement , N=250)
        kernel_q = -V_i
        #kernel_e = -tf.reduce_sum( * self._shell_displacement , axis=1)
        kernel_e = V_di - V_i
        return kernel_q, kernel_e
