import tensorflow as tf
import numpy as np
import scipy.constants as constants
import math
import functions.helping_functions as help_fn


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
        self._gaussian_width = tf.reshape(gaussian_width, [self._n_atoms, -1])
        #self._gaussian_width_a = tf.convert_to_tensor(gaussian_width[:,0],dtype=tf.float32) 
        #self._gaussian_width_b = tf.convert_to_tensor(gaussian_width[:,1],dtype=tf.float32) 
        self._sqrt_pi = tf.sqrt(pi)
        

        self.volume = tf.abs(tf.linalg.det(self._cell))

        #structure properties
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
                                  1.0 / self.volume * 
                                  tf.exp(-gmax**2*b_norm**2*sigma_min**2/2.) 
                                  / (gmax*b_norm))
                                  #1.0 / self.volume * tf.exp(-gmax**2*b_norm**2*sigma_min**2/2.) / (gmax**2*b_norm**2))
        return tf.while_loop(c, b, [gmax, result])[0]
    
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
     
    @tf.function
    def recip_space_term(self): 
        '''
        calculates the interaction contribution to the electrostatic energy for standard qeq
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

        g_width = tf.reshape(self._gaussian_width, [-1])
        N = self._n_atoms
        alpha_ij2 = g_width[:,None]**2 + g_width[None,:]**2
        g_sq = self.g_norm * self.g_norm  # [K]
        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        exp_ij_aa = (tf.exp(-0.25 * alpha_ij2[:,:,None] * g_sq[None,None,:]) / 
                     (g_sq[None,None,:] + 1e-12)) # [N,N,K]
        # The cosine term: shape [N*N,K]
        g_vecs_transpose = tf.transpose(self.g_vecs)
        #rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        rij = self._positions[:,None,:] - self._positions[None,:,:] #N,N,3
        r_dot_k = tf.matmul(rij, g_vecs_transpose)
        exp_cos_ij_aa = exp_ij_aa * tf.cos(r_dot_k) # N,N,K
        exp_sin_ij_aa = exp_ij_aa * tf.sin(r_dot_k) # N,N,K
        Vij_qq = 2.0 * tf.reduce_sum(exp_cos_ij_aa, axis=2)  # [N, N]
        #alway summing over K including the spin degeneracy
        Vija = 8.0 * tf.matmul(exp_sin_ij_aa, self.g_vecs)
        g_ab = tf.reshape(self.g_vecs[:, :, None] * self.g_vecs[:, None, :], [-1,9]) # K,3,3
        Vijab = 8.0 * tf.reshape(tf.matmul(exp_cos_ij_aa, g_ab), [N,N,3,3]) # N,N,3,3
        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        return [Vij_qq * conversion_fact,
                Vija * conversion_fact,
                Vijab * conversion_fact]
   
    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld_quadratic_qd_n(self,
                                                  z_charge):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        N = self._n_atoms
        g_width_0 = self._gaussian_width[:,0]
        g_width_ns = self._gaussian_width
        ns = tf.shape(g_width_ns)[1]
        
        alpha_ij2 = g_width_0[:,None]**2 + g_width_0[None,:]**2
        alpha_ijn2 = g_width_0[:,None,None]**2 + g_width_ns[None,:,:]**2
        alpha_ijnn2 = g_width_ns[:,None,:,None]**2 + g_width_ns[None,:,None,:]**2

        g_sq = self.g_norm * self.g_norm  # [K]
        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        exp_ij_aa = (tf.exp(-0.25 * alpha_ij2[:,:,None] * g_sq[None,None,:]) / 
                     (g_sq[None,None,:] + 1e-12)) # [N,N,K]
        exp_ij_aan = (tf.exp(-0.25 * alpha_ijn2[...,None] * g_sq[None,None,None,:]) / 
                     (g_sq[None,None,None,:] + 1e-12)) # [N,N,n,K]
        exp_ij_aann = (tf.exp(-0.25 * alpha_ijnn2[...,None] * g_sq[None,None,None,None,:]) / 
                     (g_sq[None,None,None,None,:] + 1e-12)) # [N,N,n,n,K]

        # The cosine term: shape [N*N,K]
        g_vecs_transpose = tf.transpose(self.g_vecs)
        #rij = tf.reshape(self._positions[:,None,:] - self._positions[None,:,:], [-1,3])
        rij = self._positions[:,None,:] - self._positions[None,:,:] #N,N,3
        r_dot_k = tf.matmul(rij, g_vecs_transpose)
        exp_cos_ij_aa = exp_ij_aa * tf.cos(r_dot_k) # N,N,K
        exp_sin_ij_aan = exp_ij_aan * tf.sin(r_dot_k)[:,:,None,:] # N,N,n,K
        exp_cos_ij_aann = exp_ij_aann * tf.cos(r_dot_k)[:,:,None,None,:] # N,N,n,n,K

        Vij_qq = 2.0 * tf.reduce_sum(exp_cos_ij_aa, axis=2)  # [N, N]
        #alway summing over K
        Vijna = 8.0 * tf.reshape(tf.matmul(exp_sin_ij_aan, self.g_vecs),[N,N,-1]) # N,N,n,3

        g_ab = tf.reshape(self.g_vecs[:, :, None] * self.g_vecs[:, None, :], [-1,9]) # K,9
        Vijnnab = 8.0 * tf.reshape(
                tf.transpose(
                    tf.reshape(tf.matmul(exp_cos_ij_aann, g_ab), [N,N,ns,ns,3,3]), 
                    perm=[0,1,2,4,3,5]),
                [N,N,3*ns,3*ns]) # N,N,nshells*3,nshell*3

        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        return [Vij_qq * conversion_fact,
                Vijna * conversion_fact,
                Vijnnab * conversion_fact]

    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def structure_factors(self, v):
        '''
        calculates the structure factors
        '''
        N = self._n_atoms
        g_sq = self.g_norm * self.g_norm  # [K]
        alpha2 = self._gaussian_width * self._gaussian_width

        g_vecs_transpose = tf.transpose(self.g_vecs)
        r_dot_k = tf.matmul(self._positions, g_vecs_transpose)

        fik_real = tf.exp(-0.25 * alpha2[:,None] * g_sq[None,:]) * tf.cos(r_dot_k)
        fik_im = tf.exp(-0.25 * alpha2[:,None] * g_sq[None,:]) * tf.sin(r_dot_k)

        Sk_real = tf.reduce_sum(fik_real * v[:,None], axis=0)
        Sk_im = tf.reduce_sum(fik_im * v[:,None], axis=0)
        return fik_real, fik_im, Sk_real, Sk_im

    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def structure_factors_pqeq(self, v):
        '''
        calculates the structure factors
        '''
        v = tf.reshape(v, [-1])
        N = self._n_atoms
        charges = v[:N]
        lagrange_mult = v[N]
        dipole_moment = tf.reshape(v[N+1:], [N,-1,3])
        n_shell = tf.shape(dipole_moment)[1]
        
        g_sq = self.g_norm * self.g_norm  # [K]
        alpha2 = self._gaussian_width[:,0] * self._gaussian_width[:,0]
        alphan2 = self._gaussian_width * self._gaussian_width # alp_in * alp_in

        g_vecs_transpose = tf.transpose(self.g_vecs) #(3,K)
        r_dot_k = tf.matmul(self._positions, g_vecs_transpose)
        pn_dot_k = tf.matmul(dipole_moment, g_vecs_transpose) # (N,n_shells,K)

        fik_real = tf.exp(-0.25 * alpha2[:,None] * g_sq[None,:]) * tf.cos(r_dot_k)
        fik_im = -tf.exp(-0.25 * alpha2[:,None] * g_sq[None,:]) * tf.sin(r_dot_k)

        S1k_real = tf.reduce_sum(fik_real * charges[:,None], axis=0)
        S1k_im = tf.reduce_sum(fik_im * charges[:,None], axis=0)
        
        fink_real = 2.0 * tf.exp(-0.25 * alphan2[:,:,None] * 
                                 g_sq[None,None,:]) * tf.sin(r_dot_k)[:,None,:]
        fink_im = 2.0 * tf.exp(-0.25 * alphan2[:,:,None] * 
                               g_sq[None,None,:]) * tf.cos(r_dot_k)[:,None,:]

        S2k_real = tf.reduce_sum(fink_real * pn_dot_k, axis=(0,1)) #(K)
        S2k_im = tf.reduce_sum(fink_im * pn_dot_k, axis=(0,1)) #(K)

        return [fik_real, fik_im, 
                S1k_real, S1k_im,
                fink_real, fink_im,
                S2k_real, S2k_im]

    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def A_matvec(self, v, E2):

        v = tf.reshape(v, [-1])
        conversion_fact = CONV_FACT * (8.* pi / self.volume)
        g_sq = self.g_norm * self.g_norm  # [K]
        N = self._n_atoms
        v1 = v[:N]
        lagrange_mult = v[N]
        fik_real, fik_im, Sk_real, Sk_im = self.structure_factors(v1)

        a_dot_v = tf.reduce_sum((fik_real * Sk_real[None,:] +
                                  fik_im * Sk_im[None,:]) / (g_sq[None,:] + 1e-12), axis=1)
        a_dot_v *= conversion_fact
        #add the E2 * q terms
        a_dot_v += E2 * v1
        a_dot_v += lagrange_mult # the lambda terms E2_i * qi + sum_j Vij qj + lambda = -E1 + E2*q0
        #include the langrange multiplier component that ensures square matrix
        #\sum_j q_j + 0 lambda = qtot
        a_dot_v = tf.concat([a_dot_v, [tf.reduce_sum(v1) + 1e-12]], axis=0)
        return a_dot_v

    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def M_inv(self, E2):
        N = self._n_atoms
        conversion_fact = CONV_FACT * (8.* pi / self.volume)
        g_sq = self.g_norm * self.g_norm  # [K]

        alpha = self._gaussian_width_a

        M_qi = 1.0 / (conversion_fact * 
                     tf.reduce_sum(tf.exp(-0.5 * alpha[:,None]**2 * g_sq[None,:])
                                   / (g_sq[None,:] + 1e-12), axis=-1) + 1e-12)
        M_qi += 1.0 / (E2 + 1e-12)
        M_inv = tf.concat([M_qi, [1.0e8]], axis=0)
        return M_inv

    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                ]
                 )

    def A_matvec_pqeq0(self, v, E2, E_d2):

        conversion_fact = CONV_FACT * (8.* pi / self.volume)
        g_sq = self.g_norm * self.g_norm  # [K]

        v = tf.reshape(v, [-1])
        N = self._n_atoms
        v1 = v[:N]
        lagrange_mult = v[N]
        dipole_moment = tf.reshape(v[N+1:],[N,-1,3])
        fik_real, fik_im, S1k_real, S1k_im, fink_real, fink_im, S2k_real, S2k_im = self.structure_factors_pqeq(v)
       #q:
        a_dot_v1 = tf.reduce_sum((fik_real * S1k_real[None,:] +
                                  fik_im * S1k_im[None,:]) / (g_sq[None,:] + 1e-12), axis=1)
        a_dot_v1 += tf.reduce_sum((fik_real * S2k_real[None,:] +
                                   fik_im * S2k_im[None,:]) /
                                  (g_sq[None,:] + 1e-12), axis=1) # N,
        a_dot_v1 *= conversion_fact
        #add the E2 * q terms
        a_dot_v1 += E2 * v1
        a_dot_v1 += lagrange_mult # the lambda terms E2_i * qi + sum_j Vij qj + lambda = -E1 + E2*q0
                                  # the lambda terms sum_j qj = total_charge
        #include the langrange multiplier component
        #\sum_j q_j + 0 lambda = qtot
        a_dot_v1 = tf.concat([a_dot_v1, [tf.reduce_sum(v1) + 1e-12]], axis=0)

        a_dot_v2 = tf.reduce_sum((fink_real * S1k_real[None,None,:] +
                                  fink_im * S1k_im[None,None,:])[:,:,:,None] * 
                                  self.g_vecs[None,None,:,:] / (g_sq[None,None,:,None] + 1e-12), axis=2)
        
        a_dot_v2 += tf.reduce_sum((fink_real * S2k_real[None,None,:] + 
                                   fink_im * S2k_im[None,None,:])[:,:,:,None] * 
                                  self.g_vecs[None,None,:,:] / (g_sq[None,None,:,None] + 1e-12), axis=2)

        a_dot_v2 *= conversion_fact
        a_dot_v2 += E_d2 * dipole_moment
        a_dot_v = tf.concat([a_dot_v1, tf.reshape(a_dot_v2, [-1])], axis=0)
        return a_dot_v
    
    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                ]
                 )
    def M_inv_pqeq0(self, E2, E_d2):

        N = self._n_atoms
       #q:
        conversion_fact = CONV_FACT * (8.* pi / self.volume)
        gvecs = self.g_vecs
        g_sq = self.g_norm * self.g_norm  # [K]

        alpha2 = self._gaussian_width[:,0] * self._gaussian_width[:,0]
        alphan2 = self._gaussian_width * self._gaussian_width
        Aii_q = conversion_fact * tf.reduce_sum(tf.exp(-0.5 * alpha2[:,None] * g_sq[None,:])
                                                     / (g_sq[None,:] + 1e-12), axis=-1)
        M_qi = 1.0 / (Aii_q + 1e-12)
        M_qi += 1.0 / (E2 + 1e-12)
        
        gvecs2 = gvecs * gvecs
        
        Aii_pin = conversion_fact * tf.reduce_sum(
                tf.exp(-0.5 * alphan2[:,:,None] * g_sq[None,None,:])[...,None] * 
            gvecs2[None,None,:,:] / (g_sq[None,None,:,None] + 1e-12), axis=2)
        
        M_pi = 1.0 / (Aii_pin + 1e-12)
        M_pi += 1.0 / (E_d2 + 1e-12)

        #the diagonal element at lambda is 0. We approxaimate 1/0 as 1/1e-12 to avoid overflow
        M_inv = tf.concat([M_qi, [1e12], tf.reshape(M_pi, [-1])], axis=0)
        return M_inv

    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def coulumb_energy(self, v):

        N = self._n_atoms
        g_sq = self.g_norm * self.g_norm  # [K]

        fik_real, fik_im, Sk_real, Sk_im = self.structure_factors(v)
        conversion_fact = CONV_FACT * (8.* pi / self.volume)
        #qq term
        E = tf.reduce_sum((Sk_real * Sk_real + Sk_im * Sk_im) / (g_sq + 1e-12))
        E *= conversion_fact
        return 0.5 * E

    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def coulumb_energy_qd(self, v):
        N = self._n_atoms
        g_sq = self.g_norm * self.g_norm  # [K]
       
        _,_, S1k_real, S1k_im, _, _, S2k_real, S2k_im = self.structure_factors_pqeq(v)

        conversion_fact = CONV_FACT * (8.* pi / self.volume)
        #qq term
        E = tf.reduce_sum((S1k_real * S1k_real + S1k_im * S1k_im) / (g_sq + 1e-12))
        #qz and zq term
        E += 2.0 * tf.reduce_sum((S1k_real * S2k_real + S1k_im * S2k_im) / (g_sq + 1e-12))
        #zz term
        #g_ab = self.g_vecs[:,None,:] * self.g_vecs[:,:,None]
        E += tf.reduce_sum((S2k_real * S2k_real + S2k_im * S2k_im) / (g_sq + 1e-12))
        E *= conversion_fact

        return 0.5 * E

    
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
