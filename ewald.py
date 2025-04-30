import tensorflow as tf
import numpy as np
from mendeleev import element
from scipy.special import erf
import scipy.constants as constants

pi = tf.constant(np.pi)


class ewald():
    '''
    A pyth on class to implement ewasld sum with gaussian charge distribution
    '''
    def __init__(self, positions, cell, gaussian_width,
                 acc_factor, species,
                 Gmax, rmax, eta):
        '''
           positions: atomic positions in Angstrom
           cell: cell dimension
           gaussian_width: gaussian width for each atom
           Gmax: maximum G for the the reciprocal summation
           rmax: cutoff distance for the short range term
           eta: width of the uniform gaussians
        '''
        self._cell = cell 
        self._positions = positions
        self.covalent_radii = [element(x).covalent_radius * 0.001 for x in species_sequence]
        #convert to gaussian width and to angstrom unit
        #multiply by  rt(2) because the gaussian is defined with
        self._gaussian_width = gaussian_width if gaussian_width else np.asarray(self.covalent_radii) 
        self.n_atoms = len(positions)
        self.sqrt_pi = math.sqrt(pi)
        self.volume = tf.reduce_sum(cell[0] * tf.linalg.cross(cell[1], cell[2]))
        self.reciprocal_cell = 2 * pi * tf.transpose(tf.linalg.inv(cell))

        # If a is not provided, use a default value
        if eta is None:
            eta = (self.n_atoms * w / (self.volume**2)) ** (1 / 6) * self.sqrt_pi

        # If the real space cutoff, reciprocal space cutoff and eta have not been
        # specified, use the accuracy and the weighting w to determine default
        # similarly as in https://doi.org/10.1080/08927022.2013.840898
        if rmax is None and Gmax is None:
            f = np.sqrt(-np.log(acc_factor))
            rmax = f / eta
            Gmax = 2 * eta * f
        elif rmax is None or Gmax is None:
            raise ValueError(
                "If you do not want to use the default cutoffs, please provide "
                "both cutoffs rmax and Gmax."
            )

        self.eta = eta
        self.eta_squared = self.eta**2
        self.Gmax = Gmax
        self.rmax = rmax

        #structure properties
        self.lattice_length = np.linalg.norm(cell,axis=1)
        _PBC = PBC if PBC else False

        if _PBC:
            struct_prop = compute_structure_props(lattice_vectors)
            volume, recip_lattice_vectors = struct_prop

            gamma_max = 1/(np.sqrt(2.0)*np.min(gaussian_width))
            _kmax = kmax if kmax else 2.* gamma_max * np.sqrt(-np.log(acc_factor))




    def real_space_term(gaussian_width,
                          coords,
                          compute_forces=False):
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
        CONV_FACT = 1e10 * constants.e / (4 * np.pi * constants.epsilon_0)

        Natoms = len(gaussian_width)

        coords = np.asarray(coords)
        if Natoms == 1:
            rij = np.zeros(3)
            gamma_all = 1./np.sqrt(2.)* 1./gaussian_width
        else:
            rij = coords[:,np.newaxis] - coords
            gamma_all = 1.0/np.sqrt(gaussian_width[:,np.newaxis]**2 + np.asarray(gaussian_width)**2)

        #compute the norm of all distances
        rij_norm = np.linalg.norm(rij, axis=-1)
        #trick to set the i==j element of the matrix to zero
        rij_norm_inv = np.nan_to_num(1 / rij_norm, posinf=0, neginf=0)

        #erf_term = np.where(rij_norm>0, erf(gamma_all*rij_norm)/rij_norm, 0)
        erf_term = erf(gamma_all*rij_norm) * rij_norm_inv
        V = erf_term.copy()

        V *=CONV_FACT
        E_diag = CONV_FACT * 2 / np.sqrt(np.pi) * 1/(np.sqrt(2.0)*gaussian_width)
        diag_indexes = np.diag_indices(Natoms)
        V[diag_indexes] = E_diag

        if compute_forces:
            #rij_tmp = np.where(rij_norm>0, 1/rij_norm**2,0)

            rij_tmp = rij_norm_inv * rij_norm_inv
            F_erf_part = erf_term * rij_tmp
            F_gaussian_part = -2.0/np.sqrt(np.pi) * gamma_all * np.exp(-gamma_all**2*rij_norm**2) * rij_tmp
            F_norm = F_erf_part + F_gaussian_part
            F_norm = np.tile(F_norm[:,:,np.newaxis], 3)
            force = rij * F_norm
            return [V, CONV_FACT*force, rij, gamma_all]
        return [V, rij, gamma_all]

