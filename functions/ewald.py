import tensorflow as tf
import numpy as np
import scipy.constants as constants
import math
#pi = tf.constant(math.pi)
pi = 3.141592653589793

constant_e = 1.602176634e-19
constant_eps = 8.8541878128e-12
CONV_FACT = 1e10 * constant_e / (4 * pi * constant_eps)

class ewald:
    '''
    A pyth on class to implement ewasld sum with gaussian charge distribution
    '''
    def __init__(self, positions, cell, n_atoms, gaussian_width,
                 accuracy, gmax,pbc=False,efield=None): 
        '''
           positions: atomic positions in Angstrom
           cell: cell dimension
           gaussian_width: gaussian width for each atom
           Gmax: maximum G for the the reciprocal summation
           rmax: cutoff distance for the short range term
        '''
        self._cell = tf.convert_to_tensor(cell, dtype=tf.float32)
        self._n_atoms = tf.cast(n_atoms, tf.int32)
        self._positions = tf.convert_to_tensor(positions, dtype=tf.float32)
        self._accuracy = accuracy
        self._gaussian_width = tf.convert_to_tensor(gaussian_width,dtype=tf.float32) # this should be learnable, maybe!
        self._sqrt_pi = tf.sqrt(pi)

        self.volume = tf.reduce_sum(cell[0] * tf.linalg.cross(cell[1], cell[2]))

        #structure properties
        self.cell_length = tf.linalg.norm(cell,axis=1)
        self.pbc = pbc if pbc else False

        if self.pbc:
            self.reciprocal_cell = 2 * pi * tf.transpose(tf.linalg.inv(cell))
            gamma_max = 1/(tf.sqrt(2.0)*tf.reduce_min(gaussian_width))
            self._gmax = gmax if gmax else 2.* gamma_max * tf.sqrt(-tf.math.log(accuracy))
        if efield is not None:
            self.efield = tf.squeeze(efield)
            self.reciprocal_cell = 2 * pi * tf.transpose(tf.linalg.inv(cell))
        else:
            self.efield = [0.,0.,0.]

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


    def recip_space_term(self): 
        '''
        calculates the interaction contribution to the electrostatic energy
        '''


        #compute the norm of all distances
        # the force on k due to atoms j is
        # rk-ri
        rij = self._positions[:,None,:] - self._positions[None,:,:]
        gamma_ij2 = self._gaussian_width[:,None]**2 + self._gaussian_width**2

        # refine kmax
        #gamma_max = tf.reduce_max(gamma_ij)
        #err = np.exp(-self.kmax**2/(4*gamma_max**2))
       # while err > acc_factor:
       #     kmax += 0.1
       #     err = np.exp(-kmax**2/(4*gamma_max**

       #         2))
       # kmax *= 1.00001

        # Build integer index grid for k-vectors
        # Estimate index ranges for each dimension
        b_norm = tf.linalg.norm(self.reciprocal_cell, axis=-1)
        b1_norm = b_norm[0]
        b2_norm = b_norm[1]
        b3_norm = b_norm[2]
        nmax1 = tf.cast(tf.math.floor(self._gmax / b1_norm), tf.int32)
        nmax2 = tf.cast(tf.math.floor(self._gmax / b2_norm), tf.int32)
        nmax3 = tf.cast(tf.math.floor(self._gmax / b3_norm), tf.int32)
        
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
        # Mask: exclude (0,0,0) and enforce |k|<=kmax
        g_norm = tf.linalg.norm(g_all, axis=1)
        nonzero = tf.reduce_all(tf.not_equal(replicas, 0), axis=1)
        mask = tf.logical_and(g_norm <= self._gmax, nonzero)
        g_vecs = tf.boolean_mask(g_all, mask)  # [K,3]
        g_norm = tf.boolean_mask(g_norm, mask) #[K,
        g_sq = g_norm*g_norm  # [K]
        

        # Prepare factors for summation
        # exp_factor[k,i] = exp(-g^2 * gamma_ij**2/2)
        # shape [K, N]
        g2_gamma2 = tf.einsum('ij,l->ijl', gamma_ij2/2.0, g_sq) # [N,N,K]
        exp_ij = tf.exp(-g2_gamma2)
        exp_ij_by_g_sq = exp_ij / (g_sq[None,None,:] + 1e-12)

        # The cosine term: shape [N,N,K]
        cos_term = tf.cos(tf.einsum('ijk, lk->ijl', rij, g_vecs))

        # Build energy contribution for each g as exp(-g^2 * gamma_ij**2/2)/g_sq[k] * cos_term[k]
        # Compute exp outer product and divide by k^2
        Vij = tf.reduce_sum(exp_ij_by_g_sq * cos_term, axis=-1)  # [N,N]

        # Multiply by prefactor (charges, 4π/V, Coulomb constant e²/4πε0 in eV·Å units) so that E = 1/2 \sum_ij Vij
        Vij *= CONV_FACT * (4.* np.pi / self.volume)
        
        return Vij 
    def sawtooth_PE(self):
        #k = n1 b1 + n2 b2 + n3 b3. We will set n1 = n2 = n3
        g = tf.reduce_sum(self.reciprocal_cell, axis=0)
        positions = self._positions
        gr = tf.math.sin(tf.einsum('ij,j->ij', positions, g))
        kernel = tf.einsum('ij,j,j->i', gr, self.efield, 1.0 / (g + 1e-12))
        return kernel

        
