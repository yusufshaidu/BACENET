import tensorflow as tf
import numpy as np
import scipy.constants as constants
import math
from functions.solver import newton_solve_spd
from functions.solver_general import newton_solve
import functions.helping_functions as help_fn
from functions.compute_Rij_with_mic import pairwise_vectors_with_images
#pi = tf.constant(math.pi)
pi = 3.141592653589793

constant_e = 1.602176634e-19
constant_eps = 8.8541878128e-12
CONV_FACT = 1e10 * constant_e / (4 * pi * constant_eps)
SIGMA = 0.05
N_fourier_comps = 10000
P_CUTOFF = 6.0 #This defines the cutoff to for the local dipole moments

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
        

        g_sq = self.g_norm * self.g_norm  # [K]

        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        #g2_gamma2 = tf.einsum('ij,l->ijl', sigma_ij2/4.0, g_sq) # [N*N,K]
        #chunk_size = 128
        #Vij = self.accumulate_over_g(rij, sigma_ij2, g_vecs, g_sq, chunk_size)
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
    def recip_space_term_with_shelld_linear(self,shell_displacement):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        #save the position of the shell
        shell_displacement = tf.cast(shell_displacement, tf.float32)
        #positions_plus_d = self._positions + shell_displacement
        #compute the norm of all distances
        # the force on k due to atoms j is

        # rk-ri for core-core, shell-core and shell-shell

        
        #sigma_ij2 = tf.reshape(self._gaussian_width[:,None]**2 + self._gaussian_width**2, [-1])
        sigma_ij2 = self._gaussian_width[:,None]**2 + self._gaussian_width[None,:]**2
        
        g_sq = self.g_norm * self.g_norm  # [K]
        #The g = 0 term is already removed
        #smallest_g_idx = tf.where(self.g_norm == tf.reduce_min(self.g_norm))[0]
        #smallest_g = self.g_norm[smallest_g_idx[0]] 
        #smallest_g_vec = self.g_vecs[smallest_g_idx[0]]

        #smallest_k_dir = tf.ones(3) / tf.sqrt(3.0)
        #smallest_k_dir = tf.constant([0.001,0.0,0.0])
        #small_k_dir = tf.constant([0.001,0.001,0.0]) / tf.sqrt(2.0)
        small_k_dir = tf.constant([0.001,0.001,0.001]) / tf.sqrt(3.0)
        #small_k_dir = tf.constant([0.0,0.0,0.001])
        small_k = 0.001
        #small_k_dir /= small_k

        #smallest_k_dir_dot_d = tf.matmul(shell_displacement, smallest_k_dir[:,None]) # N
        small_k_dir_dot_d = tf.reduce_sum(shell_displacement * small_k_dir[None, :], axis=1) # N


        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        #g2_gamma2 = tf.einsum('ij,l->ijl', sigma_ij2/4.0, g_sq) # [N*N,K]
        #exp_ij = tf.exp(-g2_gamma2)
        exp_ij = tf.exp(-0.25 * sigma_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        #exp_ij_by_g_sq = exp_ij * 1.0 / (g_sq[None,:] + 1e-12)
        #g_sq_inv = 1.0 / (g_sq + 1e-12)

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        #rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        rij = self._positions[:,None,:] - self._positions[None,:,:] #N,N,3
        cos_term = tf.cos(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        sin_term = tf.sin(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        Vij_qq = 2.0 * tf.reduce_sum(exp_ij * cos_term, axis=2)  # [N, N]

        #rij = tf.reshape(positions_plus_d[:,None,:] - self._positions[None,:,:], [-1,3])
        #rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        #cos_term = tf.cos(rij_dot_g)
        k_dot_d = tf.matmul(shell_displacement, g_vecs_transpose) # [N,K]
        #alway summing over K

        Vij_qz =  2.0 * tf.reduce_sum(exp_ij * k_dot_d[:,None,:] * (cos_term * k_dot_d[:,None,:] * 0.5 - sin_term), axis=2)
        #Vij_qz += 2.0 * (small_k_dir_dot_d * small_k_dir_dot_d)[:,None] / small_k / small_k
        #Vij_qz -= 4.0 * tf.reduce_sum(rij * small_k_dir[None, None, :], axis=2) * small_k_dir_dot_d[:,None] / small_k / small_k # [N,N]

        Vij_zq =  2.0 * tf.reduce_sum(exp_ij * k_dot_d[None,:,:] * (cos_term * k_dot_d[None,:,:] * 0.5 + sin_term), axis=2)

        Vij_zz =  2.0 * tf.reduce_sum(exp_ij * k_dot_d[None,:,:] * k_dot_d[:,None,:] * cos_term, axis=2)
        #Vij_zz += 2.0 * small_k_dir_dot_d[:,None] * small_k_dir_dot_d[None,:] / small_k / small_k

        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        return [Vij_qq * conversion_fact,
                Vij_qz * conversion_fact,
                Vij_zq * conversion_fact, 
                Vij_zz * conversion_fact]
    @tf.function(jit_compile=False,
                 )
    def recip_space_term_with_shelld_quadratic_qd(self):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        sigma_ij2 = self._gaussian_width[:,None]**2 + self._gaussian_width[None,:]**2

        g_sq = self.g_norm * self.g_norm  # [K]

        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        exp_ij = tf.exp(-0.25 * sigma_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        #rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        rij = self._positions[:,None,:] - self._positions[None,:,:] #N,N,3
        #cos_term = tf.cos(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        #sin_term = tf.sin(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        exp_cos_ij = exp_ij * tf.cos(tf.matmul(rij, g_vecs_transpose)) # N,N,K
        exp_sin_ij = exp_ij * tf.sin(tf.matmul(rij, g_vecs_transpose)) # N,N,K
        Vij_qq = 2.0 * tf.reduce_sum(exp_cos_ij, axis=2)  # [N, N]

        #alway summing over K
        Vija_qz =  -4.0 * tf.reduce_sum(exp_sin_ij[:,:,:,None] * 
                                        self.g_vecs[None,None, :, :], axis=2)

        g_ab = self.g_vecs[:, :, None] * self.g_vecs[:, None, :] # K,3,3
        Vijab_zz =  2.0 * tf.reduce_sum(exp_cos_ij[:,:,:, None,None] * g_ab[None,None,:,:,:], axis=2) # N,N,3,3
        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        return [Vij_qq * conversion_fact,
                Vija_qz * conversion_fact,
                Vijab_zz * conversion_fact]
     

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld_linear_Vj(self,shell_displacement,
                                               z_charge, 
                                               q_charge):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        #save the position of the shell
        shell_displacement = tf.cast(shell_displacement, tf.float32)
        #positions_plus_d = self._positions + shell_displacement
        #compute the norm of all distances
        # the force on k due to atoms j is

        # rk-ri for core-core, shell-core and shell-shell

        
        #sigma_ij2 = tf.reshape(self._gaussian_width[:,None]**2 + self._gaussian_width**2, [-1])
        #sigma_ij2 = self._gaussian_width[:,None]**2 + self._gaussian_width[None,:]**2
        
        g_sq = self.g_norm * self.g_norm  # [K]

        # Prepare factors for summation
        exp_i = tf.exp(-0.25 * self._gaussian_width[:,None]**2 * g_sq[None,:]) / (g_sq[None,:] + 1e-12) # [N,K]
        # The cosine term: shape [N*N,K]
        g_vecs_transpose = tf.transpose(self.g_vecs)

        rij = self._positions[:,None,:] - self._positions[None,:,:] #N,N,3
        cos_term = tf.cos(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        sin_term = tf.sin(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]


        Vij_qq = 2.0 * tf.reduce_sum(exp_i[:,None,:] * cos_term, axis=2)  # [N, N]
        k_dot_d = tf.matmul(shell_displacement, g_vecs_transpose) # [N,K]
        #alway summing over K
        Vij_qz =  -2.0 * tf.reduce_sum(exp_i[:,None,:] * k_dot_d[:,None,:] * sin_term, axis=2)
        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        
        #include external potentials from field
        Vj_ext = 0.0 
        #Vj_ext -= self.sawtooth_Vofr(self._positions + shell_displacement, N_fourier_comps) # electrons

        return (tf.reduce_sum(Vij_qq * q_charge[:,None], axis=0) * conversion_fact +
                tf.reduce_sum(Vij_qz * z_charge[:,None], axis=0) * conversion_fact + Vj_ext)

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

        
        sigma_ij2 = tf.reshape(self._gaussian_width[:,None]**2 + self._gaussian_width**2, [-1])
        
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

        #g2_gamma2 = 0.25 * tf.einsum('i,j->ij',sigma_ij2, g_sq) # [N*N,K]
        #exp_ij = tf.exp(-g2_gamma2)
        exp_ij = tf.exp(-0.25 * sigma_ij2[:,None] * g_sq[None,:]) # [N*N,K]
        #exp_ij_by_g_sq = exp_ij * 1.0 / (g_sq[None,:] + 1e-12)
        g_sq_inv = 1.0 / (g_sq + 1e-12)

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        #rij_dot_g = tf.matmul(rij, g_vecs_transpose) # [N*N,K]
        #cos_term = tf.cos(tf.matmul(rij, g_vecs_transpose)) #[N*N,K]
        Vij_qq = 2.0 * tf.reduce_sum(exp_ij * tf.cos(tf.matmul(rij, g_vecs_transpose)) * g_sq_inv[None,:], axis=-1)  # [N*N, K]

        rij = tf.reshape(positions_plus_d[:,None,:] - self._positions[None,:,:], [-1,3])
        #rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        #cos_term = tf.cos(rij_dot_g)
        _Vij_dj_zq =  2.0 * tf.reduce_sum(exp_ij * tf.cos(tf.matmul(rij, g_vecs_transpose)) * g_sq_inv[None,:], axis=-1)
        Vij_zq = (Vij_qq - _Vij_dj_zq)  # [N*N]

        rij = tf.reshape(self._positions[:,None,:] - positions_plus_d[None,:,:], [-1,3])
        #rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        #cos_term = tf.cos(rij_dot_g)
        _Vij_dj_qz =  2.0 * tf.reduce_sum(exp_ij * tf.cos(tf.matmul(rij, g_vecs_transpose)) * g_sq_inv[None,:], axis=-1)
        Vij_qz = (Vij_qq - _Vij_dj_qz)  # [N*N]

        rij = tf.reshape(positions_plus_d[:,None,:] - positions_plus_d[None,:,:], [-1,3])
        #rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        #cos_term = tf.cos(rij_dot_g)
        Vij_zz =  (Vij_qq  - _Vij_dj_qz - _Vij_dj_zq  + 
                   2.0 * tf.reduce_sum(exp_ij * tf.cos(tf.matmul(rij, g_vecs_transpose)) * g_sq_inv[None,:], axis=-1))  # [N*N]

        #cos_term = tf.cos(tf.reduce_sum(rij[:,:,None,:] * g_vecs[None,None,:,:], axis=-1))

        # Build energy contribution for each g as exp(-g^2 * gamma_ij**2/4)/g_sq[k] * cos_term[k]
        # Compute exp outer product and divide by k^2
        #Vij_qq = 2.0 * tf.einsum('ij,ij,j->i',exp_ij, cos_term, g_sq_inv)  # [N*N]
        #Vij_qz = 2.0 * tf.einsum('ij,ij,j->i',exp_ij, cos_term - cos_plus_dj_term, g_sq_inv)  # [N*N]
        #Vij_zz = 2.0 * tf.einsum('ij,ij,j->i',exp_ij, cos_term - 2 * cos_plus_dj_term + cos_plus_didj_term, g_sq_inv)  # [N*N]
        #divide Vij_qq by 2 because of the factor of 2 multiplied above

        # Multiply by prefactor (charges, 4pi/V, Coulomb constant e^2/4piepsilon_0 in eV-Å units) so that E = 1/2 \sum_ij Vij
        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        Vij_qq = tf.reshape(Vij_qq * conversion_fact, [self._n_atoms, self._n_atoms])
        Vij_qz = tf.reshape(Vij_qz * conversion_fact, [self._n_atoms, self._n_atoms])
        Vij_zq = tf.reshape(Vij_zq * conversion_fact, [self._n_atoms, self._n_atoms])
        Vij_zz = tf.reshape(Vij_zz * conversion_fact, [self._n_atoms, self._n_atoms])
        
        return [Vij_qq, Vij_qz, Vij_zq, Vij_zz]
    
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
        sigma_ij2 = tf.reshape(self._gaussian_width[:,None]**2 + self._gaussian_width**2, [-1])
        

        g_sq = self.g_norm * self.g_norm  # [K]

        exp_ij = tf.exp(-0.25 * sigma_ij2[:,None] * g_sq[None,:]) # [N*N,K]
        #exp_ij_by_g_sq = exp_ij * 1.0 / (g_sq[None,:] + 1e-12)
        g_sq_inv_g_alp = self.g_vecs / (g_sq[:,None] + 1e-12) + 1e-6 # [K, 3]

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        rij = tf.reshape(self._positions[:,None,:] - positions_plus_d[None,:,:], [-1,3])
        #rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        #Fij will be used through to avoid storing too many intermediate large tensors
        Fij = 2.0 * tf.matmul(exp_ij * tf.sin(tf.matmul(rij, g_vecs_transpose)), 
                              g_sq_inv_g_alp) #[N**2, 3]
  
        rij = tf.reshape(positions_plus_d[:,None,:] - self._positions[None,:,:], [-1,3])
        #rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        #sin_term = tf.sin(rij_dot_g)
        #sin_term_exp = exp_ij * sin_term
        Fij -=  tf.matmul(exp_ij * tf.sin(tf.matmul(rij, g_vecs_transpose)), 
                              g_sq_inv_g_alp) #[N**2, 3]

        zq = tf.expand_dims(z_charge[:,None] * q_charge[None,:], axis=-1)
        Fi_qz = 0.5 * tf.reduce_sum(tf.reshape(-Fij, [self._n_atoms, self._n_atoms, 3]) * zq, axis=1)

        rij = tf.reshape(positions_plus_d[:,None,:] - positions_plus_d[None,:,:], [-1,3])
        #rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        #sin_term = tf.sin(rij_dot_g)
        #sin_term_exp = exp_ij * sin_term
        #Vij_zz  =  [N^2,K] x [K,3]
        Fij = (4.0 * tf.matmul(exp_ij * tf.sin(tf.matmul(rij, g_vecs_transpose)), 
                              g_sq_inv_g_alp) - Fij) #[N**2, 3]

        zz = tf.expand_dims(z_charge[:,None] * z_charge[None,:], axis=-1)
        #terms = [N,N,3] x [N,N]
        Fi_zz = 0.5 * tf.reduce_sum(tf.reshape(Fij, [self._n_atoms, self._n_atoms, 3]) * zz, axis=1) #[N,3]

        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        F_di = Ei * shell_displacement + (Fi_qz + Fi_zz) * conversion_fact

        # compute jacobian
        #dagonal i=j
        g_sq_inv_g_alp_beta = tf.reshape(g_sq_inv_g_alp[:,:,None] * self.g_vecs[:,None,:] + 1e-6, [-1,9]) #[K,3,3]
        #exp_ii = tf.exp(-0.5 * (self._gaussian_width**2)[:,None] * g_sq[None,:]) # [N,K]
        #exp_ii = tf.exp(-g2_gamma2) # [N,K]
        #ri_dot_g = tf.matmul(shell_displacement, g_vecs_transpose) # [N,K]
        cos_kdi_exp = (tf.cos(tf.matmul(shell_displacement, g_vecs_transpose)) * 
                       tf.exp(-0.5 * (self._gaussian_width**2)[:,None] * g_sq[None,:])) #[N,K]

        charges2 = z_charge * (z_charge + q_charge) 
        # Identity matrix delta_ab
        delta = tf.eye(3)   # shape (3,3)
        # Expand and multiply
        J = tf.reshape(Ei[:, :, None] * delta[None, :, :], [-1,9])
        J += 2.0 * conversion_fact * tf.matmul(cos_kdi_exp, g_sq_inv_g_alp_beta) * charges2[:,None] # [N,9]
        
        #in the presence of field
        Fia_efield = self.sawtooth_potential_fourier_derivative(self._positions, z_charge, N=N_fourier_comps)
        J += tf.reshape(Fia_efield[:,:,None] * tf.eye(3)[None :, :], [-1,9])

        #'''
        ############ off diagonal i != j

        #cos_term = tf.cos(tf.matmul(rij, g_vecs_transpose))
        cos_term_exp = exp_ij * tf.cos(tf.matmul(rij, g_vecs_transpose))
        Fij_zz = 2.0 * conversion_fact * tf.reshape(tf.matmul(cos_term_exp, g_sq_inv_g_alp_beta), 
                                                    [self._n_atoms,self._n_atoms,3,3]) * zz[:,:,:,None] # [N, N,3,3]
        delta_ij = tf.eye(self._n_atoms)
        Jijab = tf.reshape(J, [-1,3,3])[:,None,:,:] * delta_ij[:,:,None,None] + Fij_zz * (1.0 - delta_ij[:,:,None,None]) # [N,N,3,3]
        
        #'''
        #Jiab = tf.reshape(J, [-1,3,3]) + 1.0e-6
        #we transpose Jijab to Jiajb
        return tf.reshape(F_di, [self._n_atoms*3]), tf.reshape(tf.transpose(Jijab,[0,2,1,3]), [self._n_atoms*3,self._n_atoms*3])
        #return tf.reshape(F_di, [self._n_atoms, 3]), tf.reshape(Jiab,[self._n_atoms,3,3])
        #return tf.reshape(F_di, [self._n_atoms,3,1]), J
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                ]
                 )
    def shell_optimization_newton(self, x, z_charge, q_charge, Ei,
                                   max_iter=1, tol=1e-6):
        
        # xnew = xold - \sum_{alpha} J_{i\alpha\beta} F_{i\beta} 
        '''
        def cond(i, x, dx):
            return tf.logical_and(
                tf.less(i, max_iter),
                tf.greater(tf.norm(dx), tol)
                )

        def body(i, x, dx):
            Fx, J = self.recip_space_term_with_opt_shell_func(z_charge, q_charge, Ei, x)
            #A = tf.matmul(tf.transpose(J), J)
            #B = tf.matmul(tf.transpose(J), Fx[:,None])
            #dx = tf.reshape(tf.linalg.solve(A + 1e-6 * tf.eye(3*self._n_atoms), -B),[-1,3])
            #need to regularize jacobian
            #dx = tf.reshape(tf.linalg.solve(J + 1e-3 * tf.eye(3)[None,:,:], -Fx[:,:,None]),[-1,3])
            dx = tf.reshape(tf.linalg.solve(J + 1e-3 * tf.eye(self._n_atoms*3), -Fx[:,None]),[-1,3])
            x_new = x + dx
            return i + 1, x_new, dx

        #i = tf.constant(0, dtype=tf.int32)
        i = 0
        dx = tf.ones_like(x, tf.float32)

        _, x_final, _ = tf.while_loop(cond, body, [i, x, dx])
        '''
        #we are doing only a single step for now
        Fx, J = self.recip_space_term_with_opt_shell_func(z_charge, q_charge, Ei, x)
        dx = tf.reshape(tf.linalg.solve(J + 1e-3 * tf.eye(self._n_atoms*3), -Fx[:,None]),[-1,3])
        return x + dx

        #return x_final

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_opt_shell_funct_linear(self, z_charge, q_charge, Ei):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        #compute the norm of all distances
        # the force on k due to atoms j is
        # rk-ri for core-core, shell-core and shell-shell
        #The g = 0 term is already removed
        #smallest_g_idx = tf.where(self.g_norm == tf.reduce_min(self.g_norm))[0]
        #smallest_g = self.g_norm[smallest_g_idx[0]]
        #smallest_g_vec = self.g_vecs[smallest_g_idx[0]]

        #smallest_k_dir = 1e-3 * smallest_g_vec / smallest_g
        #smallest_k_dir = tf.ones(3) / tf.sqrt(3.0)
        #small_k_dir = tf.constant([0.001,0.001,0.0]) / tf.sqrt(2.0)
        small_k_dir = tf.constant([0.001,0.001,0.001]) / tf.sqrt(3.0)
        #small_k_dir = tf.constant([0.0,0.0,0.001])
        small_k = 0.001


        sigma_ij2 = tf.reshape(self._gaussian_width[:,None]**2 + self._gaussian_width**2,[-1])
        g_sq = self.g_norm * self.g_norm  # [K]

        exp_ij = tf.exp(-0.25 * sigma_ij2[:,None] * g_sq[None,:]) # [N*N,K]
        g_sq_inv_g_alp = self.g_vecs / (g_sq[:,None] + 1e-12) # [K, 3]

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        #rij_dot_g = tf.matmul(rij, g_vecs_transpose)
        #sin_term = tf.sin(rij_dot_g)
        #cos_term = tf.cos(rij_dot_g)
        
        #sin_term_exp = exp_ij * sin_term
        #Vij_qz  =  [N^2,K] x [K,3]
        qz = tf.expand_dims(z_charge[None, :] * q_charge[:,None], axis=-1)
        zz = tf.expand_dims(z_charge[:,None] * z_charge[None,:], axis=-1)
        
        Fia_l1 = -2.0 * tf.reduce_sum(tf.reshape(tf.matmul(exp_ij * tf.sin(tf.matmul(rij, g_vecs_transpose)), 
                              g_sq_inv_g_alp),[self._n_atoms, self._n_atoms, 3]) * qz, axis=1) #[N,3]

        g_sq_inv_g_alp_beta = tf.reshape(g_sq_inv_g_alp[:,:,None] * self.g_vecs[:,None,:], [-1,9]) #[K,3*3]
        Fiab_l2 = 2.0 * tf.reduce_sum(tf.reshape(tf.matmul(exp_ij * tf.cos(tf.matmul(rij, g_vecs_transpose)),
                                                           g_sq_inv_g_alp_beta),[self._n_atoms, self._n_atoms, 3, 3]) * qz[:,:,:,None], axis=1) #[N,3,3]
        
        Fijab_l3 = 2.0 * tf.reshape(tf.matmul(exp_ij * tf.cos(tf.matmul(rij, g_vecs_transpose)),
                                              g_sq_inv_g_alp_beta),[self._n_atoms, self._n_atoms, 3, 3]) * zz[:,:,:,None] #[N,N,3,3]
        qi_sum_z = q_charge * tf.reduce_sum(z_charge)
        g_ab = (small_k_dir[:,None] * small_k_dir[None,:]) / small_k / small_k
        #non-analytic corrections
        Fiab_l2 += 2.0 * qi_sum_z[:,None,None] * g_ab[None,:,:] # N,3 3
        #Fiab_l2 += Fiab_l2_NA

        Fijab_l3 += 2.0 * zz[:,:,:,None] * g_ab[None,None,:,:] # N,N,3,3

        Fia_l1 -= 2.0 * tf.reduce_sum(
                tf.reshape(rij, [self._n_atoms, self._n_atoms, 3]) * 
                small_k_dir[None,None,:] * qz, axis=(1,2))[:,None] * small_k_dir[None,:] / small_k / small_k


        n_i, n_j, n_a, n_b = self._n_atoms, self._n_atoms, 3, 3

        #delta_ij = tf.eye(n_i)[:, :, None, None]    # (i, j, 1, 1)
        #delta_ab = tf.eye(n_a)[None, None, :, :]    # (1, 1, a, b)
        # in external electric field, the contribution is just -Zi * efield 
        Fia_0 = self.efield[None,:] * z_charge[:,None] #N,3
        
        conversion_fact = CONV_FACT * (4.* pi / self.volume)

        Fia_l1 *= conversion_fact

        Fiab_l2 *= conversion_fact
        Fia_l1 += Fia_0 #constant in d_ia
        Fiab_l2 += Ei[:,:,None] * tf.eye(3)[None :, :] # Ei in shape = (N,3)

        term1 = Fiab_l2[:,None, :,:] * tf.eye(n_i)[:, :, None, None] # [N, N, 3, 3]
        #term2 = tf.linalg.diag(Ei)[:,None,:,:] * tf.eye(n_i)[:,:,None,None] #[N,N,3,3]

        Fij = Fijab_l3 * conversion_fact + term1
        #before = [N,N,3,3]
        #after = [N,3,N,3]
        Fij = tf.reshape(tf.transpose(Fij, [0,2,1,3]), [self._n_atoms*3, self._n_atoms*3]) # Fiajb

        #Fij = tf.reshape(term1, [self._n_atoms*3, self._n_atoms*3]) - term2
        
        #J = tf.reshape(tf.linalg.diag(Ei), [-1,9]) 
        return Fij, tf.reshape(Fia_l1, [self._n_atoms*3])

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
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ]
                 )
    def compute_weights(self, first_atom_idx, 
                        second_atom_idx, 
                        central_atom_id,
                        atomic_numbers):

        atomic_numbers = tf.gather(atomic_numbers, first_atom_idx)

        central_atoms_index = (atomic_numbers == central_atom_id)
        neighbors_idx = second_atom_idx[central_atoms_index]
        atom_idx = first_atom_idx[central_atoms_index]
    
        unique_id, inverse_idx, unique_counts = tf.unique_with_counts(neighbors_idx)
        unique_id_central, inverse_idx_central, _ = tf.unique_with_counts(atom_idx)

        N_central = tf.shape(unique_id_central)[0]
        #N_all = tf.shape(unique_id)[0] + N_central
        tf_full = tf.range(self._n_atoms)

        # Build pairwise equality mask between tf_full and neighbors_idx
        eq_mask = tf.cast(tf.equal(tf_full[:, None], neighbors_idx[None, :]), tf.float32)  # [N_all, n_pairs]

        # For each pair, associate its central-atom column
        col_ids = inverse_idx_central  # [n_pairs]
        scatter_mask = tf.transpose(eq_mask)  # [n_pairs, N_all]

        # Now unsorted_segment_sum over pairs sharing the same central atom
        weights_uc_T = tf.math.unsorted_segment_sum(scatter_mask, col_ids, N_central)  # [N_central, N_all]
        weights_uc = tf.reshape(tf.transpose(weights_uc_T), [self._n_atoms,N_central])  # [N_all, N_central]
        #unique_counts are the weights for each unique_id. 
        #We can use the inverse_idx to brocast back to the original index of the central atom neighbors
        weights = tf.gather(unique_counts, inverse_idx)
        weights = 1.0 / tf.cast(weights, tf.float32) # this should be greater than 0
        return central_atoms_index, weights, weights_uc

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ]
                 )
    def polarization_linearized_periodic_component(self, Rij, 
                                          shell_displacement, 
                                          z_charge, q_charge,
                                         first_atom_idx,
                                         second_atom_idx,
                                         central_atom_id,
                                         atomic_numbers):
        '''this is based on sin fourier series of a linear potential in the 
        P_ia{\sum_{j in Neigh(i)} qj Rij - Zj dj}
        '''
        #compute weights
        central_atoms_index, weights, _ = self.compute_weights(first_atom_idx,
                                         second_atom_idx,
                                         central_atom_id,
                                         atomic_numbers)

        shell_displacement = tf.cast(shell_displacement, dtype=tf.float32)
        shell_disp_j = tf.gather(shell_displacement, second_atom_idx)[central_atoms_index]
        z_charge_j = tf.gather(z_charge, second_atom_idx)[central_atoms_index]
        q_charge_j = tf.gather(q_charge, second_atom_idx)[central_atoms_index]
        Rij = Rij[central_atoms_index]

        all_rij_norm = tf.linalg.norm(Rij, axis=-1) #npair
        mask = all_rij_norm < P_CUTOFF
        mask_float = tf.cast(mask, tf.float32)
        #Rij = tf.where(tf.tile(mask[:,None],[1,3]), Rij, tf.zeros_like(Rij))
        #shell_disp_j = tf.where(tf.tile(mask[:,None],[1,3]), shell_disp_j, tf.zeros_like(shell_disp_j))
        #q_charge_j = tf.where(mask, q_charge_j, tf.zeros_like(q_charge_j))
        #z_charge_j = tf.where(mask, z_charge_j, tf.zeros_like(z_charge_j))
        Rij *= mask_float[:,None]
        shell_disp_j *= mask_float[:,None]
        q_charge_j *= mask_float
        z_charge_j *= mask_float

        qz_charge = z_charge_j + q_charge_j

        num_segments, inverse_idx = tf.unique(first_atom_idx[central_atoms_index])
        num_segments = tf.shape(num_segments)[0]

        Pi_q = tf.math.unsorted_segment_sum(data=qz_charge[:,None] * Rij * weights[:,None],
                                            segment_ids=inverse_idx,
                                            num_segments=num_segments)
        Pi_z = -tf.math.unsorted_segment_sum(data=z_charge_j[:,None] * (Rij + shell_disp_j) * weights[:,None],
                                            segment_ids=inverse_idx,
                                            num_segments=num_segments)

        # we should add the self correction in the main function
        i_idx = (atomic_numbers == central_atom_id)
        positions_i = self._positions[i_idx]
        shell_disp_i = shell_displacement[i_idx]
        z_charge_i = z_charge[i_idx]

        P_self = -z_charge_i[:,None] * shell_disp_i

        return (tf.reshape(Pi_q, [-1,3]), 
                tf.reshape(Pi_z + P_self, [-1,3]))  #shape (Nats,3)

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ]
                 )
    def polarization_linearized_periodic(self, Rij, 
                                          shell_displacement, 
                                          z_charge, q_charge,
                                         first_atom_idx,
                                         second_atom_idx,
                                         central_atom_id,
                                         atomic_numbers):
        '''this is based on sin fourier series of a linear potential in the 
        P_ia{\sum_{j in Neigh(i)} qj Rij - Zj dj}
        '''
        #compute weights
        central_atoms_index, weights, _ = self.compute_weights(first_atom_idx,
                                         second_atom_idx,
                                         central_atom_id,
                                         atomic_numbers)

        shell_displacement = tf.cast(shell_displacement, dtype=tf.float32)
        shell_disp_j = tf.gather(shell_displacement, second_atom_idx)[central_atoms_index]
        z_charge_j = tf.gather(z_charge, second_atom_idx)[central_atoms_index]
        q_charge_j = tf.gather(q_charge, second_atom_idx)[central_atoms_index]
        Rij = Rij[central_atoms_index]

        all_rij_norm = tf.linalg.norm(Rij, axis=-1) #npair
        mask = all_rij_norm < P_CUTOFF
        mask_float = tf.cast(mask, tf.float32)
        #Rij = tf.where(tf.tile(mask[:,None],[1,3]), Rij, tf.zeros_like(Rij))
        #shell_disp_j = tf.where(tf.tile(mask[:,None],[1,3]), shell_disp_j, tf.zeros_like(shell_disp_j))
        #q_charge_j = tf.where(mask, q_charge_j, tf.zeros_like(q_charge_j))
        #z_charge_j = tf.where(mask, z_charge_j, tf.zeros_like(z_charge_j))
        Rij *= mask_float[:,None]
        shell_disp_j *= mask_float[:,None]
        q_charge_j *= mask_float
        z_charge_j *= mask_float

        qz_charge = z_charge_j + q_charge_j
        num_segments, inverse_idx = tf.unique(first_atom_idx[central_atoms_index])
        num_segments = tf.shape(num_segments)[0]

        Pi_q = tf.math.unsorted_segment_sum(data=qz_charge[:,None] * Rij * weights[:,None],
                                            segment_ids=inverse_idx,
                                            num_segments=num_segments)
        Pi_z = -tf.math.unsorted_segment_sum(data=z_charge_j[:,None] * (Rij + shell_disp_j) * weights[:,None],
                                            segment_ids=inverse_idx,
                                            num_segments=num_segments)

        # we should add the self correction in the main function
        i_idx = (atomic_numbers == central_atom_id)
        positions_i = self._positions[i_idx]
        shell_disp_i = shell_displacement[i_idx]
        z_charge_i = z_charge[i_idx]

        P_self = -z_charge_i[:,None] * shell_disp_i
        return tf.reshape(Pi_q + Pi_z + P_self, [-1,3]) #shape (Nats,3)
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ]
                 )
    def potential_linearized_periodic(self,Rij,
                                      z_charge, 
                                      atomic_numbers, 
                                      central_atom_id,
                                      first_atom_idx,
                                      second_atom_idx):
        '''this is based on sin fourier series of a linear potential in the 
        dE/dq_k = -\sum_{ij in Neigh(i)} delta_jk Rij Wij
        dE/dd_ka = \sum_{ij in Neigh(i)} delta_jk Wij efield_a
        '''
        central_atoms_index, weights, _ = self.compute_weights(first_atom_idx,
                                         second_atom_idx,
                                         central_atom_id,
                                         atomic_numbers)

        z_charge_j = tf.gather(z_charge, second_atom_idx)[central_atoms_index]
        Rij = Rij[central_atoms_index]

        all_rij_norm = tf.linalg.norm(Rij, axis=-1) #npair
        mask = all_rij_norm < P_CUTOFF
        mask_float = tf.cast(mask, tf.float32)

        Rij *= mask_float[:,None]
        z_charge_j *= mask_float

        num_segments, inverse_idx = tf.unique(first_atom_idx[central_atoms_index])
        num_segments = tf.shape(num_segments)[0]

        #delta_ij = tf.eye(num_rows=tf.shape(inverse_idx)[0], num_columns=self._n_atoms)
        delta_jk = tf.cast(tf.equal(tf.range(self._n_atoms)[None, :],
                                    second_atom_idx[central_atoms_index][:,None]), tf.float32)


        Vi_alpha = -tf.math.unsorted_segment_sum(Rij[:,None,:] * weights[:,None,None] * delta_jk[:,:,None], 
                                                      inverse_idx, num_segments=num_segments) # N_central, 
        Vi = tf.reduce_sum(tf.reduce_sum(Vi_alpha, axis=0) * self.efield[None,:], axis=1) # N
        
        dVi_alpha = tf.math.unsorted_segment_sum(weights[:,None] * delta_jk * z_charge_j[:,None], 
                                                      inverse_idx, num_segments=num_segments)
        dVi_alpha = tf.reduce_sum(dVi_alpha, axis=0) # N,3

        z_charge_i = tf.where(atomic_numbers==central_atom_id, 
                              z_charge, tf.zeros_like(z_charge))
        dVi = (dVi_alpha + z_charge_i)[:,None] * self.efield[None,:] # N,3

        return Vi, dVi #shape (Nats,3)

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ]
                 )
    def total_energy_linearized_periodic(self, Rij,
                                          shell_displacement,
                                          z_charge, q_charge,
                                          first_atom_idx,
                                         second_atom_idx,
                                         central_atom_id,
                                         atomic_numbers):
        '''this is based on sin fourier series of a linear potential in the 
        P_ia{\sum_{j in Neigh(i)} qj Rij - Zj dj}
        E = -efiled * P
        '''
        #compute weights
        central_atoms_index, weights, _ = self.compute_weights(first_atom_idx,
                                         second_atom_idx,
                                         central_atom_id,
                                         atomic_numbers)

        shell_displacement = tf.cast(shell_displacement, dtype=tf.float32)
        shell_disp_j = tf.gather(shell_displacement, second_atom_idx)[central_atoms_index]
        z_charge_j = tf.gather(z_charge, second_atom_idx)[central_atoms_index]
        q_charge_j = tf.gather(q_charge, second_atom_idx)[central_atoms_index]
        Rij = Rij[central_atoms_index]

        all_rij_norm = tf.linalg.norm(Rij, axis=-1) #npair
        mask = all_rij_norm < P_CUTOFF
        mask_float = tf.cast(mask, tf.float32)
        #Rij = tf.where(tf.tile(mask[:,None],[1,3]), Rij, tf.zeros_like(Rij))
        #shell_disp_j = tf.where(tf.tile(mask[:,None],[1,3]), shell_disp_j, tf.zeros_like(shell_disp_j))
        #q_charge_j = tf.where(mask, q_charge_j, tf.zeros_like(q_charge_j))
        #z_charge_j = tf.where(mask, z_charge_j, tf.zeros_like(z_charge_j))

        Rij *= mask_float[:,None]
        shell_disp_j *= mask_float[:,None]
        q_charge_j *= mask_float
        z_charge_j *= mask_float

        qz_charge = z_charge_j + q_charge_j
        num_segments, inverse_idx = tf.unique(first_atom_idx[central_atoms_index])
        num_segments = tf.shape(num_segments)[0]

        Pi_q = tf.math.unsorted_segment_sum(data=qz_charge[:,None] * Rij * weights[:,None],
                                            segment_ids=inverse_idx,
                                            num_segments=num_segments)
        Pi_z = -tf.math.unsorted_segment_sum(data=z_charge_j[:,None] * (Rij + shell_disp_j) * weights[:,None],
                                            segment_ids=inverse_idx,
                                            num_segments=num_segments)

        # we should add the self correction in the main function
        i_idx = (atomic_numbers == central_atom_id)
        positions_i = self._positions[i_idx]
        shell_disp_i = shell_displacement[i_idx]
        z_charge_i = z_charge[i_idx]
        P_self = -z_charge_i[:,None] * shell_disp_i

        return -tf.reduce_sum((Pi_q + Pi_z + P_self) * self.efield[None,:]) #shape (Nats,3)
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def potential_linearized_periodic_ref0(self, z_charge):
        '''this is based on sin fourier series of a linear potential in the
        P_ia = {\sum_{j} qj Rj - Zj dj}
        '''
        Vi = -tf.reduce_sum(self._positions * self.efield[None,:], axis=1)
        dVi = z_charge[:,None] * self.efield[None,:]
        return Vi, dVi #shape (Nats,3)


    ##########
    #Sawtooth potentials
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def sawtooth_potential_fourier_linearized_qd(self, z_charge):
        '''this is based on sin fourier series of a linear potential in the 
        '''
        N = N_fourier_comps
        sigma = SIGMA
        G = 2.0 * pi / self.cell_length

        counts = tf.range(N+1, dtype=tf.float32)
        coeffs = 2 * counts + 1
        g_dot_r = self._positions * G[None,:]
        smoothing = tf.exp(-0.5 * (G[:,None] * coeffs[None,:] * sigma)**2) #[3, ncoeffs]

        terms_sin = tf.math.sin(coeffs[None,None,:] * g_dot_r[:,:,None])
        terms_cos = tf.math.cos(coeffs[None,None,:] * g_dot_r[:,:,None])

        terms_cos *= ((-1.0) ** (counts)[None,None,:] * smoothing[None,:,:] / coeffs[None,None,:])
        terms_sin_1 = terms_sin * ((-1.0) ** (counts)[None,None,:] * smoothing[None,:,:] / coeffs[None,None,:]**2)
        terms_sin_2 = terms_sin * ((-1.0) ** (counts)[None,None,:] * smoothing[None,:,:])
        # implement Efield
        efield = self.efield[None,:]  # (None,3)
        efield_G = (self.efield * G)[None,:]  # (None,3)
        efield_over_G = (self.efield / G)[None,:]  # (None,3)
        
        counts2 = coeffs * coeffs
        G2 = G * G
        alpha2 = self._gaussian_width * self._gaussian_width
        args = 0.25 * counts2[None,None,:]  * G2[None,:,None] * alpha2[:,None,None]
        exp = tf.exp(-args)
        
        Vi = -4.0 / pi * tf.reduce_sum(terms_sin_1 * efield_over_G[:,:,None] * exp, axis=(1,2)) # N,
        Vi_z = 4.0 / pi * z_charge[:,None] * tf.reduce_sum(terms_cos * efield[:,:,None] * exp, axis=2) # N,
        Vi_zd = -4.0 / pi * z_charge[:,None] * tf.reduce_sum(terms_sin_2 * efield_G[:,:,None] * exp, axis=2) # N,

        return Vi, Vi_z, Vi_zd # shape ([Nats, ],[N,3],[N,3])

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def _sawtooth_potential_fourier_linearized_qd(self, z_charge):
        '''this is based on sin fourier series of a linear potential in the 
        '''
        N = N_fourier_comps
        sigma = SIGMA
        G = 2.0 * pi / self.cell_length

        counts = tf.range(1, N+1, dtype=tf.float32)
        coeffs = counts
        g_dot_r = self._positions * G[None,:]
        smoothing = tf.exp(-0.5 * (G[:,None] * coeffs[None,:] * sigma)**2) #[3, ncoeffs]

        terms_sin = tf.math.sin(coeffs[None,None,:] * g_dot_r[:,:,None])
        terms_cos = tf.math.cos(coeffs[None,None,:] * g_dot_r[:,:,None])

        terms_cos *= ((-1.0) ** (counts)[None,None,:] * smoothing[None,:,:])
        terms_sin_1 = terms_sin * ((-1.0) ** (counts)[None,None,:] * smoothing[None,:,:] / coeffs[None,None,:])
        terms_sin_2 = terms_sin * ((-1.0) ** (counts)[None,None,:] * smoothing[None,:,:] * coeffs[None,None,:])
        # implement Efield
        efield = self.efield[None,:]  # (None,3)
        efield_G = (self.efield * G)[None,:]  # (None,3)
        efield_GG = (self.efield * G * G)[None,:]  # (None,3)
        
        counts2 = coeffs * coeffs
        G2 = G * G
        alpha2 = self._gaussian_width * self._gaussian_width
        args = 0.25 * counts2[None,None,:]  * G2[None,:,None] * alpha2[:,None,None]
        exp = tf.exp(-args)
        
        Vi = 2.0 / pi * tf.reduce_sum(terms_sin_1 * efield[:,:,None] * exp, axis=(1,2)) # N,
        Vi_z = -2.0 / pi * z_charge[:,None] * tf.reduce_sum(terms_cos * efield_G[:,:,None] * exp, axis=2) # N,
        Vi_zd = 2.0 / pi * z_charge[:,None] * tf.reduce_sum(terms_sin_2 * efield_GG[:,:,None] * exp, axis=2) # N,

        return Vi, Vi_z, Vi_zd # shape ([Nats, ],[N,3],[N,3])



