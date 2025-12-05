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
P_CUTOFF = 3.5 #This defines the cutoff to for the local dipole moments

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

        self._gaussian_width_a = tf.convert_to_tensor(gaussian_width[:,0],dtype=tf.float32) 
        self._gaussian_width_b = tf.convert_to_tensor(gaussian_width[:,1],dtype=tf.float32) 
        self._gaussian_width = tf.identity(self._gaussian_width_a) # defined for  standard qeq and for obtaining kmax
        self._sqrt_pi = tf.sqrt(pi)
        

        self.volume = tf.abs(tf.linalg.det(self._cell))

        #structure properties
        self.cell_length = tf.linalg.norm(self._cell,axis=1)
        self.pbc = pbc if pbc else False

        if self.pbc:
            self.reciprocal_cell = 2 * pi * tf.transpose(tf.linalg.inv(self._cell))
            gamma_max = 1/(tf.sqrt(2.0)*tf.reduce_min(self._gaussian_width_a))
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
                                  1.0 / self.volume * tf.exp(-gmax**2*b_norm**2*sigma_min**2/2.) / (gmax*b_norm))
                                  #1.0 / self.volume * tf.exp(-gmax**2*b_norm**2*sigma_min**2/2.) / (gmax**2*b_norm**2))
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
        rij = self._positions[None,:,:] - self._positions[:,None,:]
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
        #g2_gamma2 = 0.25 * tf.einsum('i,j->ij',sigma_ij2, g_sq) # [N*N,K]
        #exp_ij = tf.exp(-g2_gamma2)
        # shape [K, N]
        exp_ij = tf.exp(-0.25 * sigma_ij2[:,None] * g_sq[None,:]) / (g_sq[None,:] + 1e-12) # [N,N,K]
        #exp_ij_by_g_sq = exp_ij * 1.0 / (g_sq[None,:] + 1e-12)
        #g_sq_inv = 1.0 / (g_sq + 1e-12)

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        rij_dot_g = tf.matmul(rij, tf.transpose(self.g_vecs))
        cos_term = tf.cos(rij_dot_g)
        #cos_term = tf.cos(tf.reduce_sum(rij[:,:,None,:] * g_vecs[None,None,:,:], axis=-1))

        # Build energy contribution for each g as exp(-g^2 * gamma_ij**2/4)/g_sq[k] * cos_term[k]
        # Compute exp outer product and divide by k^2
        Vij = 2.0 * tf.einsum('ij,ij->i',exp_ij, cos_term)  # [N,N]
        
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
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld_quadratic_qd(self,
                                                  z_charge):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        alpha_ij2 = self._gaussian_width_a[:,None]**2 + self._gaussian_width_a[None,:]**2
        beta_ij2 = self._gaussian_width_b[:,None]**2 + self._gaussian_width_b[None,:]**2
        alpha_beta_ij2 = self._gaussian_width_a[:,None]**2 + self._gaussian_width_b[None,:]**2
        #beta_alpha_ij2 = self._gaussian_width_b[:,None]**2 + self._gaussian_width_a[None,:]**2

        g_sq = self.g_norm * self.g_norm  # [K]

        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        exp_ij_aa = tf.exp(-0.25 * alpha_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        exp_ij_bb = tf.exp(-0.25 * beta_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        exp_ij_ab = tf.exp(-0.25 * alpha_beta_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        #exp_ij_ba = tf.exp(-0.25 * beta_alpha_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        exp_ij_ba = tf.transpose(exp_ij_ab, [1,0,2])

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        #rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        rij = self._positions[:,None,:] - self._positions[None,:,:] #N,N,3
        #cos_term = tf.cos(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        #sin_term = tf.sin(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        r_dot_k = tf.matmul(rij, g_vecs_transpose)
        exp_cos_ij_aa = exp_ij_aa * tf.cos(r_dot_k) # N,N,K
        exp_cos_ij_bb = exp_ij_bb * tf.cos(r_dot_k) # N,N,K
        exp_sin_ij_ab = exp_ij_ab * tf.sin(r_dot_k) # N,N,K
        exp_sin_ij_ba = exp_ij_ba * tf.sin(r_dot_k) # N,N,K
        
        Vij_qq = 2.0 * tf.reduce_sum(exp_cos_ij_aa, axis=2)  # [N, N]

        #alway summing over K
        Vija_qz =  2.0 * tf.reduce_sum(exp_sin_ij_ab[:,:,:,None] * 
                                        self.g_vecs[None,None, :, :], axis=2) * z_charge[None,:,None]
        Vija_zq =  -2.0 * tf.reduce_sum(exp_sin_ij_ba[:,:,:,None] * 
                                       self.g_vecs[None,None, :, :], axis=2) * z_charge[:,None,None]

        zz = z_charge[None,:] * z_charge[:, None] #N,N
        g_ab = self.g_vecs[:, :, None] * self.g_vecs[:, None, :] # K,3,3
        Vijab_zz =  2.0 * tf.reduce_sum(exp_cos_ij_bb[:,:,:, None,None] * g_ab[None,None,:,:,:], axis=2) # N,N,3,3
        Vijab_zz *= zz[:,:,None,None]

        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        return [Vij_qq * conversion_fact,
                Vija_qz * conversion_fact,
                Vija_zq * conversion_fact,
                Vijab_zz * conversion_fact]

    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld_quadratic_qqdd_1(self,
                                                  z_charge):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        alpha_ij2 = self._gaussian_width_a[:,None]**2 + self._gaussian_width_a[None,:]**2
        beta_ij2 = self._gaussian_width_b[:,None]**2 + self._gaussian_width_b[None,:]**2
        alpha_beta_ij2 = self._gaussian_width_a[:,None]**2 + self._gaussian_width_b[None,:]**2
        #beta_alpha_ij2 = self._gaussian_width_b[:,None]**2 + self._gaussian_width_a[None,:]**2

        g_sq = self.g_norm * self.g_norm  # [K]

        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        exp_ij_aa = tf.exp(-0.25 * alpha_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        exp_ij_bb = tf.exp(-0.25 * beta_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        exp_ij_ab = tf.exp(-0.25 * alpha_beta_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        #exp_ij_ba = tf.exp(-0.25 * beta_alpha_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        exp_ij_ba = tf.transpose(exp_ij_ab, [1,0,2])

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        #rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        rij = self._positions[:,None,:] - self._positions[None,:,:] #N,N,3
        #cos_term = tf.cos(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        #sin_term = tf.sin(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        r_dot_k = tf.matmul(rij, g_vecs_transpose)
        exp_cos_ij_aa = exp_ij_aa * tf.cos(r_dot_k) # N,N,K
        exp_cos_ij_bb = exp_ij_bb * tf.cos(r_dot_k) # N,N,K
        exp_sin_ij_ab = exp_ij_ab * tf.sin(r_dot_k) # N,N,K
        exp_cos_ij_ab = exp_ij_ab * tf.cos(r_dot_k) # N,N,K
        exp_sin_ij_ba = exp_ij_ba * tf.sin(r_dot_k) # N,N,K
        exp_cos_ij_ba = exp_ij_ba * tf.cos(r_dot_k) # N,N,K
        
        Vij_qq = 2.0 * tf.reduce_sum(exp_cos_ij_aa, axis=2)  # [N, N]

        #alway summing over K
        Vija_qz =  2.0 * tf.reduce_sum(exp_sin_ij_ab[:,:,:,None] * 
                                        self.g_vecs[None,None, :, :], axis=2) * z_charge[None,:,None]
        Vija_zq =  -2.0 * tf.reduce_sum(exp_sin_ij_ba[:,:,:,None] * 
                                       self.g_vecs[None,None, :, :], axis=2) * z_charge[:,None,None]
        
        g_ab = self.g_vecs[:, :, None] * self.g_vecs[:, None, :] # K,3,3

        Vijab_qz =  tf.reduce_sum(exp_cos_ij_ab[:,:,:,None, None] * 
                                  g_ab[None,None, :, :,:], axis=2) * z_charge[None,:,None,None]
        Vijab_zq =  tf.reduce_sum(exp_cos_ij_ba[:,:,:,None,None] * 
                                         g_ab[None,None,:,:,:], axis=2) * z_charge[:,None,None,None]

        zz = z_charge[None,:] * z_charge[:, None] #N,N
        Vijab_zz =  2.0 * tf.reduce_sum(exp_cos_ij_bb[:,:,:, None,None] * g_ab[None,None,:,:,:], axis=2) # N,N,3,3
        Vijab_zz *= zz[:,:,None,None]

        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        return [Vij_qq * conversion_fact,
                Vija_qz * conversion_fact,
                Vija_zq * conversion_fact,
                Vijab_zz * conversion_fact,
                Vijab_qz * conversion_fact,
                Vijab_zq * conversion_fact]

    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld_quadratic_qqdd_2(self,
                                                  z_charge):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        alpha_ij2 = self._gaussian_width_a[:,None]**2 + self._gaussian_width_a[None,:]**2
        beta_ij2 = self._gaussian_width_b[:,None]**2 + self._gaussian_width_b[None,:]**2
        alpha_beta_ij2 = self._gaussian_width_a[:,None]**2 + self._gaussian_width_b[None,:]**2
        #beta_alpha_ij2 = self._gaussian_width_b[:,None]**2 + self._gaussian_width_a[None,:]**2

        g_sq = self.g_norm * self.g_norm  # [K]

        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        exp_ij_aa = tf.exp(-0.25 * alpha_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        exp_ij_bb = tf.exp(-0.25 * beta_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        exp_ij_ab = tf.exp(-0.25 * alpha_beta_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        #exp_ij_ba = tf.exp(-0.25 * beta_alpha_ij2[:,:,None] * g_sq[None,None,:]) / (g_sq[None,None,:] + 1e-12) # [N,N,K]
        exp_ij_ba = tf.transpose(exp_ij_ab, [1,0,2])

        # The cosine term: shape [N*N,K]
        #cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))
        g_vecs_transpose = tf.transpose(self.g_vecs)

        #rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        rij = self._positions[:,None,:] - self._positions[None,:,:] #N,N,3
        #cos_term = tf.cos(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        #sin_term = tf.sin(tf.matmul(rij, g_vecs_transpose)) #[N,N,K]
        r_dot_k = tf.matmul(rij, g_vecs_transpose)
        exp_sin_ij_aa = exp_ij_aa * tf.sin(r_dot_k) # N,N,K
        exp_cos_ij_aa = exp_ij_aa * tf.cos(r_dot_k) # N,N,K
        exp_cos_ij_bb = exp_ij_bb * tf.cos(r_dot_k) # N,N,K
        exp_sin_ij_ab = exp_ij_ab * tf.sin(r_dot_k) # N,N,K
        exp_cos_ij_ab = exp_ij_ab * tf.cos(r_dot_k) # N,N,K
        exp_sin_ij_ba = exp_ij_ba * tf.sin(r_dot_k) # N,N,K
        exp_cos_ij_ba = exp_ij_ba * tf.cos(r_dot_k) # N,N,K
        g_ab = self.g_vecs[:, :, None] * self.g_vecs[:, None, :] # K,3,3
        
        Vij_qq = 2.0 * tf.reduce_sum(exp_cos_ij_aa, axis=2)  # [N, N]
        Vija_qq =  -2.0 * tf.reduce_sum(exp_sin_ij_aa[:,:,:,None] * 
                                        self.g_vecs[None,None, :, :], axis=2)
        Vijab_qq =  -tf.reduce_sum(exp_cos_ij_aa[:,:,:,None, None] * 
                                  g_ab[None,None, :, :,:], axis=2)

        #alway summing over K
        Vija_qz =  2.0 * tf.reduce_sum(exp_sin_ij_ab[:,:,:,None] * 
                                        self.g_vecs[None,None, :, :], axis=2) * z_charge[None,:,None]
        Vija_zq =  -2.0 * tf.reduce_sum(exp_sin_ij_ba[:,:,:,None] * 
                                       self.g_vecs[None,None, :, :], axis=2) * z_charge[:,None,None]
        

        Vijab_qz =  tf.reduce_sum(exp_cos_ij_ab[:,:,:,None, None] * 
                                  g_ab[None,None, :, :,:], axis=2) * z_charge[None,:,None,None]
        Vijab_zq =  tf.reduce_sum(exp_cos_ij_ba[:,:,:,None,None] * 
                                         g_ab[None,None,:,:,:], axis=2) * z_charge[:,None,None,None]

        zz = z_charge[None,:] * z_charge[:, None] #N,N
        Vijab_zz =  2.0 * tf.reduce_sum(exp_cos_ij_bb[:,:,:, None,None] * g_ab[None,None,:,:,:], axis=2) # N,N,3,3
        Vijab_zz *= zz[:,:,None,None]

        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        return [Vij_qq * conversion_fact,
                Vija_qz * conversion_fact,
                Vija_zq * conversion_fact,
                Vijab_zz * conversion_fact,
                Vijab_qz * conversion_fact,
                Vijab_zq * conversion_fact,
                Vija_qq * conversion_fact,
                Vijab_qq * conversion_fact]

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
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def polarization_linearized_sin(self, z_charge, q_charge, shell_disp):
        '''this is based on sin fourier series of a linear potential in the
        '''
        alph1_sq = self._gaussian_width_a * self._gaussian_width_a
        alph2_sq = self._gaussian_width_b * self._gaussian_width_b
        G = 2.0 * pi / self.cell_length
        G_sq = G * G
        positions = self._positions
        G_dot_R = G[None,:] * positions
        alpha1_g_sq = alph1_sq[:,None] * G_sq[None,:]
        alpha2_g_sq = alph2_sq[:,None] * G_sq[None,:]
        _sin_term = tf.sin(G_dot_R)
        _cos_term = tf.cos(G_dot_R)
        Piq = tf.reduce_sum(q_charge[:,None] * tf.exp(-alpha1_g_sq / 4) * 
                             _sin_term / G[None,:], axis=0)
        Pie = tf.reduce_sum(-z_charge[:,None] * 
               tf.exp(-alpha2_g_sq / 4) * 
               _cos_term * shell_disp, axis=0)
        Pie += 0.5 * tf.reduce_sum(z_charge[:,None] * tf.exp(-alpha2_g_sq / 4) * 
                _sin_term * G[None,:] * shell_disp * shell_disp, axis=0)
        return Piq, Pie

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def total_energy_sin(self, z_charge, q_charge,
                                    shell_disp):
        Piq, Pie = self.polarization_linearized_sin(z_charge, 
                                                    q_charge, 
                                                    shell_disp)
        return -tf.reduce_sum((Piq + Pie) * self._efield)

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def potential_linearized_sin(self, z_charge):
        '''this is based on sin fourier series of a linear potential in the
        P_a = {\sum_{j} qj (Rj + dj) - Zj dj}
        '''
        alph1_sq = self._gaussian_width_a * self._gaussian_width_a
        alph2_sq = self._gaussian_width_b * self._gaussian_width_b
        G = 2.0 * pi / self.cell_length
        G_sq = G * G
        positions = self._positions
        G_dot_R = G[None,:] * positions
        alpha1_g_sq = alph1_sq[:,None] * G_sq[None,:]
        _sin_term = tf.sin(G_dot_R)
        _cos_term = tf.cos(G_dot_R)
        Vi = -tf.reduce_sum(tf.exp(-alpha1_g_sq / 4) * _sin_term * self.efield[None,:] / G[None,:], axis=1)
        dVi = z_charge[:,None] * tf.exp(-alpha1_g_sq / 4) * _cos_term * self.efield[None,:]
        dVie = -z_charge[:,None] * tf.exp(-alpha1_g_sq / 4) * _sin_term * self.efield[None,:] * G[None,:]
        return Vi, dVi, dVie
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def potential_linearized_periodic_ref1(self, z_charge):
        '''this is based on sin fourier series of a linear potential in the
        P_a = {\sum_{j} qj (Rj + dj) - Zj dj}
        '''
        Vi = -tf.reduce_sum(self._positions * self.efield[None,:], axis=1)
        dVi = z_charge[:,None] * self.efield[None,:]
        dVie = -tf.tile(self.efield[None,:], [self._n_atoms,1])
        return Vi, dVi, dVie

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def potential_linearized_periodic_ref0(self, z_charge):
        '''this is based on sin fourier series of a linear potential in the
        '''
        Vi = -tf.reduce_sum(self._positions * self.efield[None,:], axis=1)
        dVi = z_charge[:,None] * self.efield[None,:]
        return Vi, dVi, tf.zeros_like(self._positions) #shape (Nats,3)
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ]
                )
    def atom_centered_dV_qs(self, 
                        z_charge, 
                        central_atom_id,
                        atomic_numbers):
        """
        This function just check that the above function is implemented correctly
        """
        cell = self._cell
        r = tf.range(-1, 2)
        X, Y, Z = tf.meshgrid(r, r, r, indexing='ij')
        replicas = tf.stack([tf.reshape(X, [-1]),
                       tf.reshape(Y, [-1]),
                       tf.reshape(Z, [-1])], axis=1)
        #R_vector = tf.matmul(replicas, cell) # shape = 27,3
        R_vector = tf.matmul(tf.cast(replicas,tf.float32), cell) # shape = 27,3

        i_idx = (atomic_numbers == central_atom_id)
        positions_i = self._positions[i_idx]
        unitcell_idx = tf.range(self._n_atoms)

        unique_type, _, counts = tf.unique_with_counts(atomic_numbers)
        composition = tf.cast(counts / tf.reduce_min(counts), tf.int32)

        #initialize P
        dViq = tf.zeros(self._n_atoms, dtype=tf.float32)
        dVie = tf.zeros((self._n_atoms,3), dtype=tf.float32)
        dViqe = tf.zeros((self._n_atoms,3), dtype=tf.float32)

        unitcell_idx = tf.tile(unitcell_idx[:,None], [1,27])
        atomic_number_rep = tf.tile(atomic_numbers[:,None], [1,27]) #shape = nat,27
        z_charge_rep = tf.tile(z_charge[:,None], [1,27]) #shape = nat,27
        position_replicas = R_vector[None,:,:] + self._positions[:,None,:] # nat, 27, 3
        #Rij = position_replicas - positions_i[:,None,:]
         
        for pos in positions_i:
            #for n_jtype, jtype in zip(composition, unique_type):
            for n_jj in tf.stack([composition, unique_type], axis=1):
                #n_jtype, jtype = n_jj
                n_jtype, jtype = n_jj[0], n_jj[1]


                idx_cast1 = (atomic_number_rep == jtype)
                Rij_jtype = position_replicas[idx_cast1] - pos
                Rij_norm_jtype = tf.linalg.norm(Rij_jtype, axis=-1)

                z_charge_jtype = z_charge_rep[idx_cast1]
                unitcell_idx_jtype = unitcell_idx[idx_cast1]

                idx_cast2 = tf.less_equal(tf.abs(Rij_norm_jtype - tf.reduce_min(Rij_norm_jtype)), 0.2)
                #idx_cast2 = (Rij_norm_jtype == tf.reduce_min(Rij_norm_jtype))
                Rij_jtype = Rij_jtype[idx_cast2] # M,3
                z_charge_jtype = z_charge_jtype[idx_cast2] #M,

                unitcell_idx_jtype = unitcell_idx_jtype[idx_cast2]
                delta_jk = tf.cast(tf.equal(tf.range(self._n_atoms)[:, None],
                                            unitcell_idx_jtype[None,:]), tf.float32) #nat,M

        
                w_jtype = tf.cast(n_jtype, tf.float32) / tf.cast(tf.shape(Rij_jtype)[0], tf.float32)


                dViq -= tf.reduce_sum(w_jtype * Rij_jtype[None,:,:] * delta_jk[:,:,None] * self.efield[None,None,:], axis=(1,2))
                dVie += tf.reduce_sum(w_jtype * (z_charge_jtype)[None,:,None] * 
                                      delta_jk[:,:,None] * self.efield[None,None,:], axis=1)
                dViqe -= tf.reduce_sum(w_jtype * delta_jk[:,:,None] * self.efield[None,None,:], axis=1)
        return dViq, dVie, dViqe

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ]
                )
    def atom_centered_total_energy(self, shell_displacement,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers):
        Piq, Pie = self.atom_centered_polarization(shell_displacement,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers)
        return -tf.reduce_sum((Piq + Pie) * self._efield)

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ]
                )
    def atom_centered_polarization_qs(self, shell_displacement,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers):

        cell = self._cell
        r = tf.range(-1, 2)
        X, Y, Z = tf.meshgrid(r, r, r, indexing='ij')
        replicas = tf.stack([tf.reshape(X, [-1]),
                       tf.reshape(Y, [-1]),
                       tf.reshape(Z, [-1])], axis=1)
        R_vector = tf.matmul(tf.cast(replicas,tf.float32), cell) # shape = 27,3

        i_idx = (atomic_numbers == central_atom_id)
        positions_i = self._positions[i_idx]

        unique_type, _, counts = tf.unique_with_counts(atomic_numbers)
        composition = tf.cast(counts / tf.reduce_min(counts), tf.int32)

        #initialize P
        Piq = tf.zeros(3, dtype=tf.float32)
        Pie = tf.zeros(3, dtype=tf.float32)

        atomic_number_rep = tf.tile(atomic_numbers[:,None], [1,27]) #shape = nat,27
        q_charge_rep = tf.tile(q_charge[:,None], [1,27]) #shape = nat,27
        z_charge_rep = tf.tile(z_charge[:,None], [1,27]) #shape = nat,27
        shell_disp_rep = tf.tile(shell_displacement[:,None,:], [1,27,1]) #shape = nat,27
        position_replicas = R_vector[None,:,:] + self._positions[:,None,:] # nat, 27, 3
        #Rij = position_replicas - positions_i[:,None,:]
         
        for pos in positions_i:
            for n_jj in tf.stack([composition, unique_type], axis=1):
                n_jtype, jtype = n_jj[0], n_jj[1]
                idx_cast1 = (atomic_number_rep == jtype)
                Rij_jtype = position_replicas[idx_cast1] - pos
                Rij_norm_jtype = tf.linalg.norm(Rij_jtype, axis=-1)

                q_charge_jtype = q_charge_rep[idx_cast1]
                z_charge_jtype = z_charge_rep[idx_cast1]
                shell_disp_rep_jtype = shell_disp_rep[idx_cast1]
                idx_cast2 = tf.less_equal(tf.abs(Rij_norm_jtype - tf.reduce_min(Rij_norm_jtype)), 0.2)
                #idx_cast2 = (Rij_norm_jtype == tf.reduce_min(Rij_norm_jtype))
                Rij_jtype = Rij_jtype[idx_cast2]
                #Rij_norm_jtype = Rij_norm_jtype[idx_cast2]
                q_charge_jtype = q_charge_jtype[idx_cast2]
                z_charge_jtype = z_charge_jtype[idx_cast2]
                shell_disp_rep_jtype = shell_disp_rep_jtype[idx_cast2]

                w_jtype = tf.cast(n_jtype, tf.float32) / tf.cast(tf.shape(Rij_jtype)[0], tf.float32)
                Piq += tf.reduce_sum(w_jtype * (z_charge_jtype)[:,None] * Rij_jtype, axis=0)
                Pie += tf.reduce_sum(w_jtype * (q_charge_jtype - z_charge_jtype)[:,None] * (Rij_jtype + shell_disp_rep_jtype), axis=0)
        return Piq, Pie

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ]
                )
    def atom_centered_polarization_vectorized(self, shell_displacement, positions,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers):

        """
        A vectorized function for polarization
        """
        cell = self._cell
        r = tf.range(-1, 2)
        X, Y, Z = tf.meshgrid(r, r, r, indexing='ij')
        replicas = tf.stack([tf.reshape(X, [-1]),
                       tf.reshape(Y, [-1]),
                       tf.reshape(Z, [-1])], axis=1)
        #R_vector = tf.matmul(replicas, cell) # shape = 27,3
        R_vector = tf.matmul(tf.cast(replicas,tf.float32), cell) # shape = 27,3

        i_idx = (atomic_numbers == central_atom_id)
        positions_i = positions[i_idx] # nat_c,3
        unitcell_idx = tf.range(self._n_atoms)

        unique_type, unique_idx, counts = tf.unique_with_counts(atomic_numbers)
        idx_cen = (unique_type == central_atom_id)
        composition = tf.cast(counts / tf.reduce_min(counts),
                                    tf.float32)
        composition_cent = composition[idx_cen]

        #initialize P
        position_replicas = R_vector[None,:,:] + positions[:,None,:] # nat, 27, 3
        Rij = position_replicas[None,...] - positions_i[:,None, None,:] # nat_c,nat,27,3

        positions_shell = positions + shell_displacement
        position_replicas_shell = R_vector[None,:,:] + positions_shell[:,None,:] # nat, 27, 3
        Rij_shell = position_replicas_shell[None,...] - positions_i[:,None, None,:] # nat_c,nat,27,3


        Rij_norm = tf.linalg.norm(Rij, axis=3) # nat_c, nat, 27
        #determine the minimum distances for each positions wrt the central atom
        min_r = tf.reduce_min(Rij_norm, axis=-1) # [nat_c, nat]
        min_at = tf.reduce_min(min_r, axis = 0) #[nat]
        #make sure to use the same minimum value across board per atoms
        min_r = tf.where(tf.abs(min_r-min_at[None,:]) < 1e-3, min_r,
                         tf.ones_like(min_r)*10000)
        #mask it out of the Rij_norm
        mask = tf.less_equal(tf.abs(Rij_norm - min_r[..., None]), 0.2)  # [nat_c,nat,27]

        # Count selected replicas per (nat_c,j). This allows to determine the wieghts 
        count_selected = tf.reduce_sum(tf.cast(mask, tf.float32), axis=-1)  # [nat_c,nat] float
          #compute the number of neighbors per atom type around a central atom.
        #In this case, we sum base on the unique id of the atoms in a simulation cell
        sum_per_type = tf.math.unsorted_segment_sum(tf.transpose(count_selected),
                                                    unique_idx, num_segments=tf.shape(unique_type)[0])
        #normalize by the composition.
        #Because of the segment sum is done per species,
        #a species with multiple atoms in a unit cell get overcounted and the weight underestimated
        sum_per_type = tf.transpose(sum_per_type) / composition[None,:] #[nat_c, nspecies]
        #broadcast to [nat_c,nat] and mask out atoms with zero neighbors
        mask_count = tf.where(count_selected > 0.0, tf.ones_like(count_selected), tf.zeros_like(count_selected))
        valid_count_selected = tf.gather(sum_per_type, unique_idx, axis=1) * mask_count
        weights_ij = tf.where(valid_count_selected > 0.0,
                              #1.0 / tf.maximum(valid_count_selected, 1.0),
                              1.0 / (valid_count_selected + 1e-10),
                              tf.zeros_like(valid_count_selected)) # to avoid division by 0

        # Sum of Rij over selected replicas -> [nat_c,nat,3]. This set all displacements with norm greater that min
        # We rewrite P_ion = \sum_c \sum_{j in uc} (q_j + Z_j)\sum_{R} (Rij + R)
        Rij_masked = tf.where(mask[..., None], Rij, tf.zeros_like(Rij))
        Rij_shell_masked = tf.where(mask[..., None], Rij_shell, tf.zeros_like(Rij_shell))

        Rij_sum = tf.reduce_sum(Rij_masked, axis=2)  # [nat_c,nat,3] # sum over periodic replicas
        # scale by the weights(the number of images per atom)
        avg_Rij = Rij_sum * weights_ij[..., None]                         # [nat_c,nat,3]
        avg_Rij = tf.where(count_selected[..., None] > 0, avg_Rij, tf.zeros_like(avg_Rij))

        Piq = tf.reduce_sum((q_charge + z_charge)[None,:,None] * 
                            avg_Rij, axis=(0,1)) / composition_cent 
        # normalization, useful if there are more than 1 central atom type per formular unit
        #computing shell contribution
        # We rewrite P_shell = -\sum_c \sum_{j in uc} Z_j\sum_{R} (Rcj + dj + R)
        Rij_shell_sum = tf.reduce_sum(Rij_shell_masked, axis=2) # sum over replicas
        avg_Rij_shell = Rij_shell_sum * weights_ij[...,None]
        Pie = -tf.reduce_sum(z_charge[None,:,None] * 
                             avg_Rij_shell, axis=(0,1)) / composition_cent
        return Piq, Pie

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                ]
                )
    def atom_centered_dV_vectorized_auto(self, shell_displacement,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers):

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(q_charge)
            tape.watch(shell_displacement)
            Piq, Pie = self.atom_centered_polarization_vectorized(
                                    shell_displacement,
                                    self._positions,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers)
            Pi_a = Piq + Pie
            energy_field = -tf.reduce_sum(Pi_a * self.efield)
        dViq = tape.gradient(energy_field, q_charge)
        dVie = tape.gradient(energy_field, shell_displacement)
        return dViq, dVie

