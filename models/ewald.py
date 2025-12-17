import tensorflow as tf
import numpy as np
import scipy.constants as constants
import math
import functions.helping_functions as help_fn
from models.coulomb_functions import (_compute_energy_exact,
                                      _compute_charges)


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
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld_exact(self, shell_displacement):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        #save the position of the shell

        shell_displacement = tf.cast(shell_displacement, tf.float32)
        shell_displacement = tf.reshape(shell_displacement, [self._n_atoms,-1,3])
        shell_positions = self._positions[:,None,:] + shell_displacement

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

        g_vecs_transpose = tf.transpose(self.g_vecs)

        rij = self._positions[:,None,:] - self._positions[None,:,:]
        cos_ric_jc = tf.cos(tf.matmul(rij, g_vecs_transpose))
        Vij_qq = 2.0 * tf.reduce_sum(exp_ij_aa * cos_ric_jc, axis=-1)  # [N, N]
        
        #core vs shell
        cos_ris_jc = tf.cos(tf.matmul(self._positions[None,:,None,:] - shell_positions[:,None,:,:], 
                                      g_vecs_transpose)) # N,N,nshells,K
        cos_ric_js = tf.cos(tf.matmul(shell_positions[None,...] - self._positions[:,None,None,:],
                                      g_vecs_transpose)) # N,N,nshells,K

        cos_ris_js = tf.cos(tf.matmul(shell_positions[None,:,None,:,:] - shell_positions[:,None,:,None,:],
                                      g_vecs_transpose)) # N,N,nshells,nshells,K

        Vij_qz = 2.0 * tf.reduce_sum(exp_ij_ab * cos_ric_jc, axis=2)
        Vijn_qz = -2.0 * tf.reduce_sum(exp_ij_ab[:,:,None,:] * cos_ric_js, axis=3)
        
        Vij_zq = 2.0 * tf.reduce_sum(exp_ij_ba * cos_ric_jc, axis=2)
        Vijn_zq = -2.0 * tf.reduce_sum(exp_ij_ba[:,:,None,:] * cos_ris_jc, axis=3)
        
        Vij_zz = 2.0 * tf.reduce_sum(exp_ij_bb * cos_ric_jc, axis=2)
        Vijn_zz1 = -2.0 * tf.reduce_sum(exp_ij_bb[:,:,None,:] * cos_ric_js, axis=3)
        Vijn_zz2 = -2.0 * tf.reduce_sum(exp_ij_bb[:,:,None,:] * cos_ris_jc, axis=3)
        Vijnn_zz = 2.0 * tf.reduce_sum(exp_ij_bb[:,:,None,None,:] * cos_ris_js, axis=-1)

        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        return [Vij_qq * conversion_fact, 
                Vij_qz * conversion_fact,
                Vijn_qz * conversion_fact,
                Vij_zq * conversion_fact,
                Vijn_zq * conversion_fact,
                Vij_zz * conversion_fact,
                Vijn_zz1 * conversion_fact,
                Vijn_zz2 * conversion_fact,
                Vijnn_zz * conversion_fact]
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld_exact_2(self, shell_displacement):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        #save the position of the shell

        shell_displacement = tf.cast(shell_displacement, tf.float32)
        #shell_displacement = tf.reshape(shell_displacement, [self._n_atoms,3])
        shell_positions = self._positions + shell_displacement

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

        g_vecs_transpose = tf.transpose(self.g_vecs)

        #core vs shell
        cos_ric_jc = tf.cos(tf.matmul(self._positions[:,None,:] - self._positions[None,:,:], 
                                      g_vecs_transpose))

        cos_ris_jc = tf.cos(tf.matmul(self._positions[None,:,:] - shell_positions[:,None,:], 
                                      g_vecs_transpose)) # N,N,K
        cos_ric_js = tf.cos(tf.matmul(shell_positions[None,...] - self._positions[:,None,:],
                                      g_vecs_transpose)) # N,N,K

        cos_ris_js = tf.cos(tf.matmul(shell_positions[None,:,:] - shell_positions[:,None,:],
                                      g_vecs_transpose)) # N,N,K

        Vij_qq = 2.0 * tf.reduce_sum(exp_ij_aa * cos_ris_js, axis=3)  # [N, N]
        Vij_qz = 2.0 * tf.reduce_sum(exp_ij_ab * (cos_ris_jc - cos_ris_js), axis=2)
        Vij_zq = 2.0 * tf.reduce_sum(exp_ij_ba * (cos_ric_js - cos_ris_js), axis=2)
        Vij_zz = 2.0 * tf.reduce_sum(exp_ij_bb * (cos_ric_jc + cos_ris_js - 
                                                  cos_ric_js - cos_ris_jc), axis=2)

        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        return [Vij_qq * conversion_fact, 
                Vij_qz * conversion_fact,
                Vij_zq * conversion_fact,
                Vij_zz * conversion_fact]
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld_exact_derivatives(self, shell_displacement, z_charge, q_charge, 
                                                       E_d2, field_kernel_e):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        #save the position of the shell

        shell_displacement = tf.cast(shell_displacement, tf.float32)
        shell_displacement = tf.reshape(shell_displacement, [self._n_atoms,-1,3])

        n_shells = tf.shape(E_d2)[1]
        z_charge_n = tf.tile(z_charge[:,None], [1,n_shells]) / tf.cast(n_shells, tf.float32)

        shell_positions = self._positions[:,None,:] + shell_displacement # N,nshell,3

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

        g_vecs_transpose = tf.transpose(self.g_vecs)

        rij = self._positions[:,None,:] - self._positions[None,:,:]
        cos_ric_jc = tf.cos(tf.matmul(rij, g_vecs_transpose))
        sin_ric_jc = tf.sin(tf.matmul(rij, g_vecs_transpose))
        Vij_qq = 2.0 * tf.reduce_sum(exp_ij_aa * cos_ric_jc, axis=-1)  # [N, N]
        #core vs shell
        cos_ris_jc = tf.cos(tf.matmul(self._positions[None,:,None,:] - shell_positions[:,None,:,:], 
                                      g_vecs_transpose)) # N,N,nshells,K
        cos_ris_js = tf.cos(tf.matmul(shell_positions[None,:,None,:,:] - shell_positions[:,None,:,None,:],
                                      g_vecs_transpose)) # N,N,nshells,nshells,K
        cos_ric_js = tf.cos(tf.matmul(shell_positions[None,...] - self._positions[:,None,None,:],
                                      g_vecs_transpose)) # N,N,nshells,K

        Vij_qz = 2.0 * tf.reduce_sum(exp_ij_ab * cos_ric_jc, axis=2)
        Vijn_qz = -2.0 * tf.reduce_sum(exp_ij_ab[:,:,None,:] * cos_ric_js, axis=3)

        Vij_zq = 2.0 * tf.reduce_sum(exp_ij_ba * cos_ric_jc, axis=2)
        Vijn_zq = -2.0 * tf.reduce_sum(exp_ij_ba[:,:,None,:] * cos_ris_jc, axis=3)


        sin_ris_jc = tf.sin(tf.matmul(self._positions[None,:,None,:] - shell_positions[:,None,:,:], 
                                      g_vecs_transpose)) # N,N,nshells,K
        sin_ris_js = tf.sin(tf.matmul(shell_positions[None,:,None,:,:] - shell_positions[:,None,:,None,:],
                                      g_vecs_transpose)) # N,N,nshells,nshells,K
        
        z_charge_ijnn = z_charge_n[:,None,:,None] * z_charge_n[None,:,None,:]
        #Fina = 2.0 * tf.einsum('ijk, ijmnk, ijmn, ka->ima',
        #                       exp_ij_bb, sin_ris_js,z_charge_ijnn,self.g_vecs)
        Fina = 2.0 * tf.reduce_sum(tf.matmul(exp_ij_bb[:,:,None,None,:] * sin_ris_js, self.g_vecs) 
                                   * z_charge_ijnn[...,None], axis=(1,3))
        z_charge_ijn = z_charge[None,:, None] * z_charge_n[:,None,:]
        #Fina -= 2.0 * tf.einsum('ijk, ijmk, ijm, ka->ima',
        #                        exp_ij_bb, sin_ris_jc,z_charge_ijn,self.g_vecs)
        Fina -= 2.0 * tf.reduce_sum(tf.matmul(exp_ij_bb[:,:,None,:] * sin_ris_jc, self.g_vecs) 
                                   * z_charge_ijn[...,None], axis=1)

        qz_charge_ijn = q_charge[None,:,None] * z_charge_n[:,None,:]
        Fina -= 2.0 * tf.reduce_sum(tf.matmul(exp_ij_ba[:,:,None,:] * sin_ris_jc, self.g_vecs) 
                                   * qz_charge_ijn[...,None], axis=1)

        #Fina -= 2.0 * tf.einsum('ijk, ijmk, ijm, ka->ima',
        #                        exp_ij_ba, sin_ris_jc,qz_charge_ijn,self.g_vecs)
        
        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        Fina *= conversion_fact
        Fina += field_kernel_e
        Fina += E_d2 * shell_displacement 

        g_ab = tf.reshape(self.g_vecs[:, :, None] * self.g_vecs[:, None, :], [-1,9]) # K,3,3
        
        #Jijnnab = 2.0 * tf.einsum('ijk, ijmnk,ijmn,kab->ijmnab', 
        #                          exp_ij_bb, cos_ris_js, z_charge_ijnn, g_ab)
        Jijnnab = 2.0 * tf.reshape(tf.matmul(exp_ij_bb[:,:,None,None,:] * cos_ris_js, g_ab) * z_charge_ijnn[...,None],
                                   [self._n_atoms, self._n_atoms,n_shells,n_shells,3,3])
        #Jinab = 2.0 * tf.einsum('ijk, ijmk, ijm, kab->imab',exp_ij_bb, cos_ris_jc, z_charge_ijn, g_ab)
        Jinab = 2.0 * tf.reduce_sum(tf.matmul(exp_ij_bb[:,:,None,:] * cos_ris_jc, g_ab) * z_charge_ijn[...,None], 
                                    axis=1)

        Jinab += 2.0 * tf.reduce_sum(tf.matmul(exp_ij_ba[:,:,None,:] * cos_ris_jc, g_ab) * qz_charge_ijn[...,None],   
                                    axis=1)
        Jinab = tf.reshape(Jinab, [self._n_atoms,n_shells,3,3])
        
        #2.0 * tf.einsum('ijk, ijmk, ijm, kab->imab',exp_ij_ba, cos_ris_jc, qz_charge_ijn, g_ab)

        Jijnnab *= conversion_fact
        Jinab *= conversion_fact
        Jinab += E_d2[:,:,:,None] * tf.eye(3)

        Jijnnab += (Jinab[:,None,:,None,:,:] * 
                    tf.eye(self._n_atoms)[:,:,None,None,None,None] * 
                    tf.eye(n_shells)[None,None,:,:,None,None])

        Fina = tf.reshape(Fina, [-1])
        dim = 3 * n_shells * self._n_atoms
        Jijnnab = tf.reshape(tf.transpose(Jijnnab, perm=(0,2,4,1,3,5)), [dim,dim])

        return (Vij_qq * conversion_fact,
                Vij_qz * conversion_fact,
                Vijn_qz * conversion_fact,
                Vij_zq * conversion_fact,
                Vijn_zq * conversion_fact,
                Fina, Jijnnab)
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld_exact_2_derivatives(self, shell_displacement, z_charge, q_charge, 
                                                       E_d2, field_kernel_e):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''

        #save the position of the shell

        shell_displacement = tf.cast(shell_displacement, tf.float32)
        shell_displacement = tf.reshape(shell_displacement, [self._n_atoms,3])

        shell_positions = self._positions + shell_displacement # N,3

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

        g_vecs_transpose = tf.transpose(self.g_vecs)
         
        #core vs shell
        cos_ric_jc = tf.cos(tf.matmul(self._positions[:,None,:] - self._positions[None,:,:], 
                                      g_vecs_transpose))

        sin_ric_jc = tf.sqrt(1.0 - cos_ric_jc * cos_ric_jc + 1e-6)
        cos_ris_jc = tf.cos(tf.matmul(self._positions[None,:,:] - shell_positions[:,None,:], 
                                      g_vecs_transpose)) # N,N,K
        sin_ris_jc = tf.sqrt(1.0 - cos_ris_jc * cos_ris_jc + 1e-6)

        cos_ric_js = tf.cos(tf.matmul(shell_positions[None,...] - self._positions[:,None,:],
                                      g_vecs_transpose)) # N,N,K
        #sin_ric_js = tf.sqrt(1.0 - cos_ric_js * cos_ric_js + 1e-6)

        cos_ris_js = tf.cos(tf.matmul(shell_positions[None,:,:] - shell_positions[:,None,:],
                                      g_vecs_transpose)) # N,N,K
        sin_ric_js = tf.sqrt(1.0 - cos_ris_js * cos_ris_js + 1e-6)

        Vij_qq = 2.0 * tf.reduce_sum(exp_ij_aa * cos_ris_js, axis=3)  # [N, N]
        Vij_qz = 2.0 * tf.reduce_sum(exp_ij_ab * (cos_ris_jc - cos_ris_js), axis=2)
        Vij_zq = 2.0 * tf.reduce_sum(exp_ij_ba * (cos_ric_js - cos_ris_js), axis=2)
        #Vij_zz = 2.0 * tf.reduce_sum(exp_ij_bb * (cos_ric_jc + cos_ris_js - 
        #                                          cos_ric_js - cos_ris_jc), axis=2)

        z_charge_ij = z_charge[:,None] * z_charge[None,:]
        q_charge_ij = q_charge[:,None] * q_charge[None,:]
        qz_charge_ij = q_charge[:,None] * z_charge[None,:]
        qz_charge_ji = tf.transpose(qz_charge_ij)

        Fina = 2.0 * tf.reduce_sum(tf.matmul(exp_ij_aa * sin_ris_js, self.g_vecs) 
                                   * q_charge_ij[...,None], axis=1)

        Fina += 2.0 * tf.reduce_sum(tf.matmul(exp_ij_bb * (sin_ris_js - sin_ris_jc), self.g_vecs) 
                                   * z_charge_ij[...,None], axis=1)

        Fina -= 2.0 * tf.reduce_sum(tf.matmul(exp_ij_ab * sin_ris_js, self.g_vecs) 
                                   * (qz_charge_ji + qz_charge_ij)[...,None], axis=1)
        Fina += 2.0 * tf.reduce_sum(tf.matmul(exp_ij_ab * sin_ris_jc, self.g_vecs) 
                                   * qz_charge_ij[...,None], axis=1)


        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        Fina *= conversion_fact
        Fina += field_kernel_e
        Fina += E_d2 * shell_displacement 

        g_ab = tf.reshape(self.g_vecs[:, :, None] * self.g_vecs[:, None, :], [-1,9]) # K,3,3
        
        Jijab = 2.0 * tf.reshape(tf.matmul(exp_ij_aa * cos_ris_js, g_ab) * q_charge_ij[...,None],
                                   [self._n_atoms, self._n_atoms,3,3])
        
        Jijab += 2.0 * tf.reshape(tf.matmul(exp_ij_bb * cos_ris_js, g_ab) * z_charge_ij[...,None],
                                   [self._n_atoms, self._n_atoms,3,3])
        


        Jinab = 2.0 * tf.reduce_sum(tf.matmul(exp_ij_bb[:,:,None,:] * cos_ris_jc, g_ab) * z_charge_ijn[...,None], 
                                    axis=1)

        Jinab += 2.0 * tf.reduce_sum(tf.matmul(exp_ij_ba[:,:,None,:] * cos_ris_jc, g_ab) * qz_charge_ijn[...,None],   
                                    axis=1)
        Jinab = tf.reshape(Jinab, [self._n_atoms,n_shells,3,3])
        
        #2.0 * tf.einsum('ijk, ijmk, ijm, kab->imab',exp_ij_ba, cos_ris_jc, qz_charge_ijn, g_ab)

        Jijnnab *= conversion_fact
        Jinab *= conversion_fact
        Jinab += E_d2[:,:,:,None] * tf.eye(3)

        Jijnnab += (Jinab[:,None,:,None,:,:] * 
                    tf.eye(self._n_atoms)[:,:,None,None,None,None] * 
                    tf.eye(n_shells)[None,None,:,:,None,None])

        Fina = tf.reshape(Fina, [-1])
        dim = 3 * n_shells * self._n_atoms
        Jijnnab = tf.reshape(tf.transpose(Jijnnab, perm=(0,2,4,1,3,5)), [dim,dim])

        return (Vij_qq * conversion_fact,
                Vij_qz * conversion_fact,
                Vijn_qz * conversion_fact,
                Vij_zq * conversion_fact,
                Vijn_zq * conversion_fact,
                Fina, Jijnnab)

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                 )
    def solver(self, E_d2, field_kernel_e, z_charge, atomic_q0,
           total_charge, b, E2, n_shells,
           tol=1e-3, max_iter=50):
    #
        N = tf.shape(atomic_q0)[0]
        z_charge_n = tf.broadcast_to(z_charge[:,None] / tf.cast(n_shells, tf.float32), [N,n_shells])
        charges0 = atomic_q0
        shell_disp = tf.ones((N, 3*n_shells)) * 0.05
        count0 = tf.constant(0)
        conv0 = False
        E0 = 0.0

        E_d2 = tf.reshape(E_d2, [N, n_shells, 3])
        field_kernel_e = tf.reshape(field_kernel_e, [N, n_shells, 3])
        def _body(shell_disp, charges, count, converged):
            bb = b
            charges_old = charges
            Vij_qq, Vij_qz, Vijn_qz, Vij_zq, Vijn_zq, dE_d_d, ddE_dd_d = \
                    self.recip_space_term_with_shelld_exact_derivatives(shell_disp, z_charge, charges, 
                                                                        E_d2, field_kernel_e)

            bb += 0.5 * tf.reduce_sum(
                    (tf.transpose(Vij_zq * z_charge[:,None], perm=(1, 0)) + Vij_qz * z_charge[None,:]),
                axis=1
            )
            bb += 0.5 * tf.reduce_sum(
                    (tf.transpose(Vijn_zq * z_charge_n[:,None,:], perm=(1, 0, 2)) + Vijn_qz * z_charge_n[None,:,:]),
                axis=(1, 2)
            )
            charges = _compute_charges(Vij_qq, bb, E2, atomic_q0, total_charge)
            #L = tf.linalg.cholesky(ddE_dd_d)
            #dshell_disp = -tf.reshape(
            #        tf.linalg.cholesky_solve(L, dE_d_d[:,:,None]),
            #        [N, 3*n_shells])
            dshell_disp = -tf.reshape(tf.linalg.solve(ddE_dd_d, dE_d_d[:,None]),
                                      [N,3*n_shells])
            converged = (tf.linalg.norm(dshell_disp) +
                         tf.linalg.norm(charges-charges_old)) < tol
            return shell_disp + dshell_disp, charges, count + 1, converged
        @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(None,None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
        def single_iter(shell_disp, charges):
            bb = tf.identity(b)
            Vij_qq, Vij_qz, Vijn_qz, Vij_zq, Vijn_zq, dE_d_d, ddE_dd_d = \
                    self.recip_space_term_with_shelld_exact_derivatives(shell_disp, z_charge, charges, 
                                                                        E_d2, field_kernel_e)
            bb += 0.5 * tf.reduce_sum(
                    (tf.transpose(Vij_zq * z_charge[:,None], perm=(1, 0)) + Vij_qz * z_charge[None,:]),
                axis=1
            )
            bb += 0.5 * tf.reduce_sum(
                    (tf.transpose(Vijn_zq * z_charge_n[:,None,:], perm=(1, 0, 2)) + Vijn_qz * z_charge_n[None,:,:]),
                axis=(1, 2)
            )

            charges = _compute_charges(Vij_qq, bb, E2, atomic_q0, total_charge)

            dshell_disp = -tf.reshape(tf.linalg.solve(ddE_dd_d, dE_d_d[:,None]),
                                      [N,3*n_shells])
            return shell_disp + dshell_disp, charges
        
        def _cond(shell_disp, charges, count, converged):
            return tf.logical_and(
                tf.logical_not(converged),
                count < max_iter
            )

        #Execute SCF
        '''
        shell_disp_final, charges, _, _ = tf.while_loop(
            _cond,
            _body,
            loop_vars=[shell_disp, charges0, count0, conv0],
            parallel_iterations = 1,
            maximum_iterations=2
        )
        '''

        '''
        for _ in range(4):
            shell_disp_final, charges = single_iter(shell_disp, charges0)
            charges0 = charges
            shell_disp = shell_disp_final
        '''
        shell_disp_final, charges = single_iter(shell_disp, charges0)
        #shell_disp_final, charges = single_iter(shell_disp_final, charges)

        _V = self.recip_space_term_with_shelld_exact(shell_disp_final)
        Vij_qq, Vij_qz, Vijn_qz, Vij_zq, Vijn_zq, Vij_zz, Vijn_zz1, Vijn_zz2, Vijnn_zz = _V

        E = _compute_energy_exact(Vij_qq, Vij_qz, Vijn_qz, Vij_zq,
                               Vijn_zq, Vij_zz, Vijn_zz1,
                               Vijn_zz2, Vijnn_zz,
                               E_d2, shell_disp_final,
                               charges, z_charge)

        return charges, shell_disp_final, E

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

        N = self._n_atoms

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
        Vija_qz = 2.0 * tf.matmul(exp_sin_ij_ab, self.g_vecs) * z_charge[None,:,None]
        Vija_zq = -2.0 * tf.matmul(exp_sin_ij_ba, self.g_vecs) * z_charge[:,None,None]

        zz = z_charge[None,:] * z_charge[:, None] #N,N
        g_ab = tf.reshape(self.g_vecs[:, :, None] * self.g_vecs[:, None, :], [-1,9]) # K,3,3
        Vijab_zz = 2.0 * tf.reshape(tf.matmul(exp_cos_ij_bb, g_ab) * 
                                     zz[...,None], [N,N,3,3]) # N,N,3,3
        conversion_fact = CONV_FACT * (4.* pi / self.volume)
        return [Vij_qq * conversion_fact,
                Vija_qz * conversion_fact,
                Vija_zq * conversion_fact,
                Vijab_zz * conversion_fact]

    @tf.function(jit_compile=False,
                 input_signature=[
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                ]
                 )
    def recip_space_term_with_shelld_quadratic_q_nd(self, n_shells, z_charge):
        '''
        calculates the interaction contribution to the electrostatic energy
        '''
        #z_charge = tf.ones((self._n_atoms, n)) * 2.0 # [nat, n]
        z_charge = tf.tile(z_charge[:,None], [1,n_shells]) / tf.cast(n_shells, tf.float32)

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
        Vija_qz =  2.0 * tf.matmul(exp_sin_ij_ab, self.g_vecs)[:,:,None,:] * z_charge[None,:,:,None] # [N,N,n,3]
        Vija_zq =  -2.0 * tf.matmul(exp_sin_ij_ba, self.g_vecs)[:,:,None,:] * z_charge[:,None,:,None] # [N,N,n,3]

        zz = z_charge[None,:,:,None] * z_charge[:, None,None,:] #N,N,n,n
        g_ab = tf.reshape(self.g_vecs[:, :, None] * self.g_vecs[:, None, :], [-1,9])
        Vijab_zz =  2.0 * tf.matmul(exp_cos_ij_bb, g_ab) # N,N,3,3
        Vijab_zz = tf.reshape(Vijab_zz[:,:,None,None,:] * zz[...,None],
                              [self._n_atoms, self._n_atoms,n_shells,n_shells,3,3]) # [N,N,n,n,3,3]

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

        N = self._n_atoms

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
        Vija_qz = 2.0 * tf.matmul(exp_sin_ij_ab, self.g_vecs) * z_charge[None,:,None]
        Vija_zq = -2.0 * tf.matmul(exp_sin_ij_ba, self.g_vecs) * z_charge[:,None,None]
        g_ab = tf.reshape(self.g_vecs[:, :, None] * self.g_vecs[:, None, :], [-1,9]) # K,3,3
        Vijab_qz = tf.reshape(tf.matmul(exp_cos_ij_ab, g_ab) * 
                               z_charge[None,:,None], [N,N,3,3])
        Vijab_zq = tf.reshape(tf.matmul(exp_cos_ij_ba, g_ab) * 
                              z_charge[:,None,None], [N,N,3,3])

        zz = z_charge[None,:] * z_charge[:, None] #N,N
        Vijab_zz =  2.0 * tf.reshape(tf.matmul(exp_cos_ij_bb, g_ab) *
                                     zz[...,None], [N,N,3,3]) # N,N,3,3

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
        N = self._n_atoms
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
        g_ab = tf.reshape(self.g_vecs[:, :, None] * self.g_vecs[:, None, :], [-1,9]) # K,9
        
        Vij_qq = 2.0 * tf.reduce_sum(exp_cos_ij_aa, axis=2)  # [N, N]
        Vija_qq = -2.0 * tf.matmul(exp_sin_ij_aa, self.g_vecs)
        Vijab_qq = -tf.reshape(tf.matmul(exp_cos_ij_aa, g_ab), [N,N,3,3])

        #alway summing over K
        Vija_qz = 2.0 * tf.matmul(exp_sin_ij_ab, self.g_vecs) * z_charge[None,:,None]
        Vija_zq = -2.0 * tf.matmul(exp_sin_ij_ba, self.g_vecs) * z_charge[:,None,None]
        
        Vijab_qz = tf.reshape(tf.matmul(exp_cos_ij_ab, g_ab) * 
                               z_charge[None,:,None], [N,N,3,3])
        Vijab_zq = tf.reshape(tf.matmul(exp_cos_ij_ba, g_ab) * 
                               z_charge[:,None,None], [N,N,3,3])
        zz = z_charge[None,:] * z_charge[:, None] #N,N
        Vijab_zz =  2.0 * tf.reshape(tf.matmul(exp_cos_ij_bb ,
                                               g_ab) * zz[...,None], [N,N,3,3]) # N,N,3,3
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
        return Vi, dVi, tf.zeros((self._n_atoms,3)) #shape (Nats,3)
    
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                )
    def atom_centered_polarization(self, shell_displacement, positions,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers,
                                    n_shells=1):

        """
        A vectorized function for polarization
        """

        n_shells = tf.shape(shell_displacement)[1]
        z_charge = tf.tile(z_charge[:,None], [1,n_shells]) / tf.cast(n_shells, tf.float32)

        #z_charge = tf.cond(tf.greater(n_shells, 1), lambda: tf.ones((self._n_atoms, n_shells)) * 2.0,
        #        lambda: z_charge[:,None])
        #if tf.greater(n_shells, 1):
        #    z_charge = tf.ones((self._n_atoms, n_shells)) * 2.0 # 2.0 for each shell
        #else:
        #    z_charge = z_charge[:,None]
        cell = self._cell
        r = tf.range(-1, 2)
        X, Y, Z = tf.meshgrid(r, r, r, indexing='ij')
        replicas = tf.stack([tf.reshape(X, [-1]),
                       tf.reshape(Y, [-1]),
                       tf.reshape(Z, [-1])], axis=1)
        #R_vector = tf.matmul(replicas, cell) # shape = 27,3
        R_vector = tf.matmul(tf.cast(replicas,tf.float32), cell) # shape = 27,3

        i_idx = (atomic_numbers == central_atom_id)
        positions_i = tf.reshape(positions[i_idx], [-1,3]) # nat_c,3
        nat_c = tf.shape(positions_i)[0]
        unitcell_idx = tf.range(self._n_atoms)

        unique_type, unique_idx, counts = tf.unique_with_counts(atomic_numbers)
        idx_cen = (unique_type == central_atom_id)
        composition = tf.cast(counts / tf.reduce_min(counts),
                                    tf.float32)
        composition_cent = composition[idx_cen]

        #initialize P
        position_replicas = R_vector[None,:,:] + positions[:,None,:] # nat, 27, 3
        Rij = position_replicas[None,...] - positions_i[:,None, None,:] # nat_c,nat,27,3

        positions_shell = positions[:,None,:] + shell_displacement # nat,nshells,3

        position_replicas_shell = R_vector[None,:,None,:] + positions_shell[:,None,:,:] # nat, 27, n_shells,3
        Rij_shell = (position_replicas_shell[None,...] - 
                               positions_i[:,None, None,None,:]) # nat_c,nat,27,n_shells,3

        Rij_norm = tf.linalg.norm(Rij, axis=3) # nat_c, nat, 27
        #compute the minimum distance per atom for each central atom and thier replicas
        # include a buffer of less the minimum distance itself. Because, if rmin_i is the distance of atom i from the central atom,
        # atoms at r < rmin are images
        min_r = tf.reduce_min(
            tf.reshape(
                tf.transpose(Rij_norm, [1,0,2]),
                       [-1,nat_c*27]),
            axis=1) * 1.8 # [nat_c, nat]
        #get a mask of valid atoms
        mask = tf.less_equal(Rij_norm, min_r[None,:,None])

        #To determin the weights, we sum over all true values
        _count_selected = tf.reduce_sum(tf.cast(mask, tf.float32), axis=-1)  # [nat_c,nat] float
        #How many centra atoms do I share
        count_selected = tf.reduce_sum(_count_selected, axis=0)[None,:] * tf.cast(_count_selected>=1, dtype=tf.float32)
        #print(count_selected, _count_selected)
        valid_count_selected = count_selected

        weights_ij = tf.math.divide_no_nan(tf.ones((nat_c,self._n_atoms)) , valid_count_selected)
        #weights_ij = tf.where(valid_count_selected > 0.0,
        #                      #1.0 / tf.maximum(valid_count_selected, 1.0),
        #                      1.0 / (valid_count_selected + 1e-10),
        #                      tf.zeros((nat_c,self._n_atoms))) # to avoid division by 0

        # Sum of Rij over selected replicas -> [nat_c,nat,3]. This set all displacements with norm greater that min
        # We rewrite P_ion = \sum_c \sum_{j in uc} (q_j + Z_j)\sum_{R} (Rij + R)
        Rij_masked = tf.where(mask[..., None], Rij, tf.zeros((nat_c,self._n_atoms, 27, 3)))
        Rij_shell_masked = tf.where(mask[..., None,None], Rij_shell, tf.zeros((nat_c,self._n_atoms, 27, n_shells,3))) # nat_c, nat,27,3*n_shells
        #Rij_shell_masked = tf.reshape(Rij_shell_masked, [-1,self._n_atoms, 27, n_shells,3])

        Rij_sum = tf.reduce_sum(Rij_masked, axis=2)  # [nat_c,nat,3*n_shells] # sum over periodic replicas
        # scale by the weights(the number of images per atom)
        avg_Rij = Rij_sum * weights_ij[..., None]                         # [nat_c,nat,3]
        #avg_Rij = tf.where(count_selected[..., None] > 0, avg_Rij, tf.zeros((nat_c,self._n_atoms, 3*n_shells)))

        Piq = tf.reduce_sum((q_charge + tf.reduce_sum(z_charge, axis=1))[None,:,None] * 
                            avg_Rij, axis=(0,1)) / composition_cent 
        # normalization, useful if there are more than 1 central atom type per formular unit
        #computing shell contribution
        # We rewrite P_shell = -\sum_c \sum_{j in uc} Z_j\sum_{R} (Rcj + dj + R)
        Rij_shell_sum = tf.reduce_sum(Rij_shell_masked, axis=2) # sum over replicas
        avg_Rij_shell = Rij_shell_sum * weights_ij[...,None,None]
        #avg_Rij_shell = tf.where(count_selected[..., None] > 0, avg_Rij_shell, tf.zeros((nat_c,self._n_atoms, 3*n_shells)))
        #z_charge = tf.reshape(z_charge[:,None,:], [self._n_atoms, 3, n_shells])

        Pie = -tf.reduce_sum(z_charge[None,:,:,None] * 
                             avg_Rij_shell, axis=(0,1)) / composition_cent # n_shells,3
        return Piq, tf.reshape(Pie, [-1,3])

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                )
    def atom_centered_dV(self, shell_displacement,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers,
                                    n_shells=1):
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(q_charge)
            tape.watch(shell_displacement)
            Piq, Pie = self.atom_centered_polarization(
                                    shell_displacement,
                                    self._positions,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers,
                                    n_shells=n_shells)
            energy_field = -(tf.reduce_sum(Piq * self.efield) + 
                             tf.reduce_sum(Pie * self.efield[None,:]))
        dViq = tape.gradient(energy_field, q_charge)
        dVie = tape.gradient(energy_field, shell_displacement)

        return dViq, tf.reshape(dVie, [-1,n_shells*3])
    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                )
    def atom_centered_polarization_2(self, shell_displacement, positions,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers,
                                    n_shells=1):

        """
        A vectorized function for polarization
        """

        z_charge = tf.tile(z_charge[:,None], [1,n_shells]) / tf.cast(n_shells, tf.float32)

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

        positions_shell = positions[:,None,:] + shell_displacement # nat,nshells,3

        position_replicas_shell = R_vector[None,:,None,:] + positions_shell[:,None,:,:] # nat, 27, 3, n_shells
        Rij_shell = tf.reshape(position_replicas_shell[None,...] - 
                               positions_i[:,None, None,None,:], [-1, self._n_atoms, 27, 3*n_shells]) # nat_c,nat,27,3*n_shells

        nat_c = tf.shape(positions_i)[0]

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
        Rij_masked = tf.where(mask[..., None], Rij, tf.zeros((nat_c,self._n_atoms, 27, 3*n_shells)))
        Rij_shell_masked = tf.where(mask[..., None], Rij_shell, tf.zeros((nat_c,self._n_atoms, 27, 3*n_shells))) # nat_c, nat,27,3*n_shells
        #Rij_shell_masked = tf.reshape(Rij_shell_masked, [-1,self._n_atoms, 27, n_shells,3])

        Rij_sum = tf.reduce_sum(Rij_masked, axis=2)  # [nat_c,nat,3*n_shells] # sum over periodic replicas
        # scale by the weights(the number of images per atom)
        avg_Rij = Rij_sum * weights_ij[..., None]                         # [nat_c,nat,3]

        #avg_Rij = tf.where(count_selected[..., None] > 0, avg_Rij, tf.zeros(shape))

        Piq = tf.reduce_sum(tf.reduce_sum(z_charge, axis=1)[None,:,None] * 
                            avg_Rij, axis=(0,1)) / composition_cent 
        # normalization, useful if there are more than 1 central atom type per formular unit
        #computing shell contribution
        # We rewrite P_shell = -\sum_c \sum_{j in uc} Z_j\sum_{R} (Rcj + dj + R)
        Rij_shell_sum = tf.reduce_sum(Rij_shell_masked, axis=2) # sum over replicas
        avg_Rij_shell = Rij_shell_sum * weights_ij[...,None,None]
        #z_charge = tf.reshape(z_charge[:,None,:], [self._n_atoms, 3, n_shells])

        Pie = tf.reduce_sum((q_charge[:,None] - z_charge)[None,:,:,None] * 
                             avg_Rij_shell, axis=(0,1)) / composition_cent # n_shells,3
        return Piq, Pie

    @tf.function(jit_compile=False,
                input_signature=[
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                ]
                )
    def atom_centered_dV_2(self, shell_displacement,
                                    z_charge, q_charge,
                                    central_atom_id,
                                    atomic_numbers,
                                    n_shells=1):
        
        with tf.GradientTape(persistent=True) as gg:
            with tf.GradientTape(persistent=True) as g:
                g.watch(q_charge)
                gg.watch(q_charge)
                g.watch(shell_displacement)
                gg.watch(shell_displacement)

                Piq, Pie = self.atom_centered_polarization_2(
                                        shell_displacement,
                                        self._positions,
                                        z_charge, q_charge,
                                        central_atom_id,
                                        atomic_numbers,
                                        n_shells=n_shells)
                energy_field = -(tf.reduce_sum(Piq * self.efield) + 
                                 tf.reduce_sum(Pie * self.efield[None,:]))
            dViq = g.gradient(energy_field, q_charge)
            dVie = g.gradient(energy_field, shell_displacement)
        dViqe = gg.jacobian(dVie, q_charge)
        # We are interested in the [i,i,:] components
        dViqe = tf.transpose(tf.reshape(dViqe, (self._n_atoms,3, self._n_atoms)), [0,2,1])
        idx = tf.stack([tf.range(self._n_atoms), 
                        tf.range(self._n_atoms)], axis=1)    # shape (N, 2)
        dViqe = tf.gather_nd(dViqe, idx)
        return dViq, tf.reshape(dVie, [self._n_atoms,n_shells*3]), tf.reshape(dViqe, [self._n_atoms,n_shells*3])

