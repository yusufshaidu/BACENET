import numpy as np
import tensorflow as tf
import mendeleev
from mendeleev import element
import math 
import itertools, os
#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
from networks.networks import Networks
import functions.helping_functions as help_fn
from functions.tf_linop import AOperator
from models.cg import ConjugateGradientSolver
from models.descriptors import (to_three_body_order_terms, to_four_body_order_terms,
                                to_five_body_order_terms, to_three_body_order_terms_chunk)

from models.coulomb_functions import (
                                      _compute_charges_disp, 
                                      _compute_charges,
                                      _compute_coulumb_energy,
                                      _compute_coulumb_energy_pqeq_qd,
                                      _compute_coulumb_energy_pqeq, 
                                      )
from models.ewald import ewald
from models.polarization import (atom_centered_polarization,
                                 atom_centered_dV)
#from models.iterative_solver import solver

import warnings
import logging
constant_e = 1.602176634e-19
epsilon_0 = 8.854e-12 # F/m
Angs = 1e-10 #m
UNIT_FACTOR = constant_e / (Angs * epsilon_0)


pi = 3.141592653589793
constant_e = 1.602176634e-19

CONV_FACT = 1e10 * constant_e / (4 * pi * epsilon_0)

#tf.config.run_functions_eagerly(
#    True
#)
#tf.debugging.enable_check_numerics()
class Compute:
    def __init__(self, configs):

        self.configs = configs
        self.setup()

        #-----------------networks
        self.exact_solver = configs['exact_solver']
        self.atomic_nets = self.configs['atomic_nets']
        self.radial_funct_net = self.configs['radial_funct_net']
        #---------------descriptors
        if self.configs['body_order'] == 3:
            self.to_body_order_terms = to_three_body_order_terms
            #self.to_body_order_terms = to_three_body_order_terms_chunk
        elif self.configs['body_order'] == 4:
            self.to_body_order_terms = to_four_body_order_terms
        elif self.configs['body_order'] == 5:
            self.to_body_order_terms = to_five_body_order_terms

        self.lxlylz = []
        self.lxlylz_sum = []
        self.fact_norm = []
        for i, z in enumerate(self.zeta):
            lxlylz, lxlylz_sum, fact_norm = help_fn._compute_cosine_terms(z)
            lxlylz = tf.cast(lxlylz,tf.float32) #[n_lxlylz, 3]
            lxlylz_sum = tf.cast(lxlylz_sum, tf.int32) #[n_lxlylz,]
            fact_norm = tf.cast(fact_norm, tf.float32) #[n_lxlylz,]
            self.lxlylz.append(lxlylz)
            self.lxlylz_sum.append(lxlylz_sum)
            self.fact_norm.append(fact_norm)

    def setup(self):
        configs = self.configs
        # -------------------------
        # === BASIC GEOMETRY ===
        # -------------------------
        self._n_shells = configs['n_shells']
        self.rcut = float(configs["rc_rad"])
        self.Nrad = int(configs["Nrad"])
        self.zeta = configs["zeta"]    # List of l_max per body order
        self.body_order = int(configs["body_order"])

        # -------------------------
        # === SPECIES INFORMATION ===
        # -------------------------
        self.species = configs["species"]
        self.nspecies = len(self.species)

        # species embedding network sizes
        self.species_layer_sizes = configs["species_layer_sizes"]
        self.nspec_embedding = int(self.species_layer_sizes[-1])

        # species correlation mode
        self.species_correlation = configs["species_correlation"]

        # size of species features per atom
        self.spec_size = int(configs["spec_size"])

        # -------------------------
        # === FEATURE DIMENSIONS ===
        # -------------------------
        self.feature_size = int(configs["feature_size"])

        # number of Bessel radial components
        self.n_bessels = int(configs.get("n_bessels", self.Nrad))

        # -------------------------
        # === DISPERSION PARAMETERS ===
        # -------------------------
        self.include_vdw = bool(configs["include_vdw"])
        self.rmin_u = float(configs["rmin_u"])
        self.rmax_u = float(configs["rmax_u"])
        self.rmin_d = float(configs["rmin_d"])
        self.rmax_d = float(configs["rmax_d"])

        self.nelement = int(configs["nelement"])
        self.linear_d_terms = configs['linear_d_terms']

        # external field
        efield = configs.get("efield", None)
        if efield is not None:
            self.efield = tf.cast(efield, tf.float32)
            self.apply_field = True
        else:
            self.efield = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
            self.apply_field = False

        print("Electric field:", self.efield.numpy())

        # -------------------------
        # === SIMULATION SETTINGS ===
        # -------------------------
        self.accuracy = configs["accuracy"]
        self.pbc = bool(configs["pbc"])
        self.central_atom_id = int(configs["central_atom_id"])
        self.gaussian_width_scale = float(configs["gaussian_width_scale"])

        # -------------------------
        # === RADIAL INDEXING ===
        # -------------------------
        self.number_radial_components = 1
        for i, z in enumerate(self.zeta):
            self.number_radial_components += (i + 1) * z + 1

    @tf.function(jit_compile=False,
                input_signature=[(
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                )])
    def tf_predict_energy_forces(self,x):
        '''
        x = (batch_species_encoder,positions,
                    nmax_diff, batch_nats,cells,C6,
                first_atom_idx,second_atom_idx,shift_vectors,num_pairs)
        '''
        rc = tf.constant(self.rcut,dtype=tf.float32)
        Nrad = tf.constant(self.Nrad, dtype=tf.int32)
        #thetasN = tf.constant(self.thetaN, dtype=tf.int32)

        nat = x[3]
        nmax_diff = x[2]

        species_encoder = tf.reshape(x[0][:nat*self.nspec_embedding],
                                     [nat,self.nspec_embedding])

        #width = tf.cast(x[1], dtype=tf.float32)
        positions = tf.reshape(x[1][:nat*3], [nat,3])
        positions = positions
        cell = tf.reshape(x[4], [3,3])
        #evdw = 0.0
        num_pairs = x[9]
        first_atom_idx = tf.cast(x[6][:num_pairs], tf.int32)
        second_atom_idx = tf.cast(x[7][:num_pairs],tf.int32)
        shift_vector = tf.cast(tf.reshape(x[8][:num_pairs*3],
                                          [num_pairs,3]), tf.float32)
        
        #shouls be removed in the future
        #gaussian_width = tf.reshape(x[10][:nat*2], [nat,2])
        #C6 = x[5][:nat]
        #chi0 = x[11][:nat]
        #J0 = x[12][:nat]
        #atomic_q0 = x[13][:nat]
        #total_charge = x[14]
        #nuclei_charge = x[15][:nat]
        #atomic_number = x[16][:nat]


        with tf.GradientTape(persistent=True) as tape0:
            tape0.watch(positions)
            tape0.watch(cell)

            #based on ase 
            #npairs x 3
            all_rij = tf.gather(positions,second_atom_idx) - \
                     tf.gather(positions,first_atom_idx) + \
                    tf.tensordot(shift_vector,cell,axes=1)

            all_rij_norm = tf.linalg.norm(all_rij, axis=-1) #npair
            reg = 1e-12
            species_encoder_i = tf.gather(species_encoder,first_atom_idx)
            species_encoder_j = tf.gather(species_encoder,second_atom_idx)
            #if self.species_correlation=='tensor':
            #    species_encoder_extended = tf.expand_dims(species_encoder_i, -1) * tf.expand_dims(species_encoder_j, -2)
            #else:
            species_encoder_extended = species_encoder_i * species_encoder_j
            species_encoder_ij = tf.reshape(species_encoder_extended, 
                                                  [-1, self.spec_size]
                                                   ) #npairs,spec_size

            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x Nneigh x embedding
            #fcuts is nat x Nneigh and reshaped to nat x Nneigh
            #_Nneigh = tf.shape(all_rij_norm)
            #neigh = _Nneigh[1]
            kn_rad = tf.ones(self.n_bessels,dtype=tf.float32)
            bf_radial0 = help_fn.bessel_function(all_rij_norm,
                                            rc,kn_rad,
                                            self.n_bessels) #

            bf_radial1 = tf.reshape(bf_radial0, [-1,self.n_bessels])
            bf_radial2 = self.radial_funct_net(bf_radial1)
            bf_radial = tf.reshape(bf_radial2, [num_pairs, self.Nrad, self.number_radial_components])
            radial_ij = tf.expand_dims(bf_radial, 2) * tf.expand_dims(tf.expand_dims(species_encoder_ij, 1), -1)
            radial_ij = tf.reshape(radial_ij, [num_pairs, self.Nrad*self.spec_size,self.number_radial_components])
            atomic_descriptors = tf.math.unsorted_segment_sum(data=radial_ij[:,:,0],
                                                              segment_ids=first_atom_idx, num_segments=nat) 

            #implement angular part: compute vect_rij dot vect_rik / rij / rik
            #rij_unit = tf.einsum('ij,i->ij',all_rij, 1.0 / (all_rij_norm+reg)) #npair,3
            rij_unit = all_rij / (tf.expand_dims(all_rij_norm + reg, -1))
            Gi = self.to_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat)
            Gi = tf.concat(Gi, axis=1)
            atomic_descriptors = tf.concat([atomic_descriptors, Gi], axis=1)
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])

            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            total_energy = tf.reduce_sum(_atomic_energies)
        #differentiating a scalar w.r.t tensors
        pad_rows = tf.zeros([nmax_diff], dtype=tf.float32)
        forces = tape0.gradient(total_energy, positions)
        #needs tape to be persistent
        dE_dh = tape0.gradient(total_energy, cell)
        V = tf.abs(tf.tensordot(cell[0], tf.linalg.cross(cell[1], cell[2]),axes=1))
            # Stress: stress_{ij} = (1/V) \sum_k dE/dh_{ik} * h_{jk} = -1/V * virial
        stress = tf.linalg.matmul(dE_dh, cell, transpose_b=True) / V
        pad_rows = tf.zeros([nmax_diff, tf.shape(forces)[1]], dtype=forces.dtype)
        forces = tf.concat([-forces, pad_rows], axis=0)
        #unused terms
        charges = tf.zeros(nat+nmax_diff)
        E1 = tf.zeros(nat+nmax_diff)
        E2 = tf.zeros(nat+nmax_diff)
        C6 = tf.zeros(nat+nmax_diff)
        Pi_a = tf.zeros(3)
        return [total_energy, forces, C6, charges, stress, Pi_a, E1, E2]

    #--------------------------------------------------
    # standard charge equilibration 
    #--------------------------------------------------

    @tf.function(jit_compile=False,
                input_signature=[(
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                )])
    def tf_predict_energy_forces_qeq(self,x):
        '''
        x = (batch_species_encoder,positions,
                    nmax_diff, batch_nats,cells,C6,
                first_atom_idx,second_atom_idx,shift_vectors,num_pairs)
        '''
        rc = tf.constant(self.rcut,dtype=tf.float32)
        Nrad = tf.constant(self.Nrad, dtype=tf.int32)
        #thetasN = tf.constant(self.thetaN, dtype=tf.int32)

        nat = x[3]
        nmax_diff = x[2]

        species_encoder = tf.reshape(x[0][:nat*self.nspec_embedding],
                                     [nat,self.nspec_embedding])

        #width = tf.cast(x[1], dtype=tf.float32)
        positions = tf.reshape(x[1][:nat*3], [nat,3])
        positions = positions
        cell = tf.reshape(x[4], [3,3])
        evdw = 0.0
        C6 = x[5][:nat]
        num_pairs = x[9]
        first_atom_idx = tf.cast(x[6][:num_pairs], tf.int32)
        second_atom_idx = tf.cast(x[7][:num_pairs],tf.int32)
        shift_vector = tf.cast(tf.reshape(x[8][:num_pairs*3],
                                          [num_pairs,3]), tf.float32)
        gaussian_width = tf.reshape(x[10][:nat*2], [nat,2])
        chi0 = x[11][:nat]
        J0 = x[12][:nat]
        atomic_q0 = x[13][:nat]
        total_charge = x[14]
        nuclei_charge = x[15][:nat]
        atomic_number = x[16][:nat]
        _efield = self.efield
        with tf.GradientTape(persistent=True) as tape0:
            tape0.watch(positions)
            tape0.watch(cell)
            tape0.watch(_efield)

            #based on ase 
            #npairs x 3
            all_rij = tf.gather(positions,second_atom_idx) - \
                     tf.gather(positions,first_atom_idx) + \
                    tf.tensordot(shift_vector,cell,axes=1)

            all_rij_norm = tf.linalg.norm(all_rij, axis=-1) #npair
            reg = 1e-12
            species_encoder_i = tf.gather(species_encoder,first_atom_idx)
            species_encoder_j = tf.gather(species_encoder,second_atom_idx)
            #if self.species_correlation=='tensor':
            #    species_encoder_extended = tf.expand_dims(species_encoder_i, -1) * tf.expand_dims(species_encoder_j, -2)
            #else:
            species_encoder_extended = species_encoder_i * species_encoder_j
            species_encoder_ij = tf.reshape(species_encoder_extended, 
                                                  [-1, self.spec_size]
                                                   ) #npairs,spec_size

            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x Nneigh x embedding
            #fcuts is nat x Nneigh and reshaped to nat x Nneigh
            #_Nneigh = tf.shape(all_rij_norm)
            #neigh = _Nneigh[1]
            kn_rad = tf.ones(self.n_bessels,dtype=tf.float32)
            bf_radial0 = help_fn.bessel_function(all_rij_norm,
                                            rc,kn_rad,
                                            self.n_bessels) #

            bf_radial = tf.reshape(bf_radial0, [-1,self.n_bessels])
            bf_radial = self.radial_funct_net(bf_radial)
            bf_radial = tf.reshape(bf_radial, [num_pairs, self.Nrad, self.number_radial_components])
            radial_ij = tf.expand_dims(bf_radial, 2) * tf.expand_dims(tf.expand_dims(species_encoder_ij, 1), -1)
            radial_ij = tf.reshape(radial_ij, [num_pairs, self.Nrad*self.spec_size,self.number_radial_components])
            atomic_descriptors = tf.math.unsorted_segment_sum(data=radial_ij[:,:,0],
                                                              segment_ids=first_atom_idx, num_segments=nat) 

            #implement angular part: compute vect_rij dot vect_rik / rij / rik
            #rij_unit = tf.einsum('ij,i->ij',all_rij, 1.0 / (all_rij_norm+reg)) #npair,3
            rij_unit = all_rij / (tf.expand_dims(all_rij_norm + reg, -1))
            Gi = self.to_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat)
            Gi = tf.concat(Gi, axis=1)
            atomic_descriptors = tf.concat([atomic_descriptors, Gi], axis=1)

            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])

            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            total_energy = tf.reduce_sum(_atomic_energies[:,0])
            idx = 1
            E1 = tf.nn.softplus(_atomic_energies[:,idx])
            idx += 1
            E2 = tf.nn.softplus(_atomic_energies[:,idx])
            #include atomic electronegativity(chi0) and hardness (J0)
            E1 += chi0
            E2 += J0

            _b = tf.identity(E1)
            _b -= E2 * atomic_q0 # only if we are not optimizing deq

            _ewald = ewald(positions, cell, nat,
                    gaussian_width,self.accuracy,
                           None, self.pbc, _efield,
                           self.gaussian_width_scale
                           )
            Vij = _ewald.recip_space_term() if self.pbc else _ewald.real_space_term()

            shell_disp = tf.zeros((nat,1,3)) # not needed
            nuclei_charge = tf.zeros(nat) # not needed, only place_holder
            if self.apply_field:
                #directly diffrentiate charge and shell_disp.
                #Since the derivative is independent on charge and shell_disp, we are initializing them to zero
                #Not: this doesn's affect the derivatives needed here

                #shell_disp = tf.zeros((nat,3))[:,None,:]
                charges = tf.identity(atomic_q0)
                #field_kernel_q, field_kernel_e, field_kernel_qe = _ewald.atom_centered_dV_2(shell_disp,
                field_kernel_q, field_kernel_e = atom_centered_dV(all_rij,shell_disp,
                                                                  nuclei_charge,charges,
                                                                  first_atom_idx,second_atom_idx,
                                                                  atomic_number, self.central_atom_id,
                                                                  _efield)
                field_kernel_qe = tf.zeros((nat,self._n_shells*3))
            else:
                field_kernel_q,field_kernel_e, field_kernel_qe = tf.zeros(nat), tf.zeros((nat,3)), tf.zeros((nat,3))

            _b += field_kernel_q

            cell_volume = tf.abs(tf.linalg.det(cell))
            if not self.exact_solver:
                #currently unstable
                #solve linear system
                @tf.function
                def A_matvec(v):
                    v = tf.reshape(v, [-1])
                    return _ewald.A_matvec(v, E2)
                    #return tf.reshape(_ewald.A_matvec(v, E2), [-1])

                b = tf.concat([-_b, [total_charge]], axis=0)
                x0 = tf.concat([atomic_q0, [1e-8]], axis=0)
                n = b.shape[0]
                linop = AOperator(A_matvec, n)
                linop_M_inv = tf.linalg.LinearOperatorDiag(
                        _ewald.M_inv(E2))
                results = tf.linalg.experimental.conjugate_gradient(
                            linop,
                            b,
                            preconditioner=linop_M_inv,
                            tol=1e-6,
                            max_iter=100
                        )
                #this is to prevect backpropagation through the cg loop. 
                #Very important to get the correct forces. 
                #This is correct because the forces do not depend on the derivative of q or p 
                # w.r.t atomic position, thanks to pqeq
                charges = tf.stop_gradient(results[1][:nat])

                #charges = ConjugateGradientSolver(A_matvec, M_inv,
                #                                       tol=1e-6,
                #                                  maxiter=1000).solve(b,x0)[:nat]

                
                ecoul = _ewald.coulumb_energy(charges)
                dq = charges - atomic_q0
                ecoul += tf.reduce_sum(E1 * dq + 0.5 * E2 * dq * dq)
            else:
                charges = _compute_charges(Vij, _b, E2, total_charge)
                ecoul = _compute_coulumb_energy(charges, atomic_q0, E1, E2, Vij)

            #if self.apply_field:
            _shell_disp = tf.reshape(shell_disp, [nat,self._n_shells,3])
            Piq_a, Pie_a = atom_centered_polarization(all_rij,_shell_disp,
                                                          nuclei_charge,charges,
                                                          first_atom_idx,second_atom_idx,
                                                          atomic_number, self.central_atom_id)
            #else:
            #    Piq_a, Pie_a = tf.zeros(3), tf.zeros((self._n_shells,3))


            Pi_a = Piq_a 
            efield_energy = -tf.reduce_sum(Piq_a * _efield)
            ecoul += efield_energy
            total_energy += ecoul

            Pi_a /= cell_volume

        #differentiating a scalar w.r.t tensors
        if self.apply_field:
            #Z*_{iab} = V * dP_a/dr_{ib}
            Zstar = cell_volume * tf.transpose(tape0.jacobian(Pi_a, positions, 
                                                              experimental_use_pfor=False), perm=(1,0,2))
            #epsilon*_{ab} = dP_a/defield_b
            epsilon_infty = tape0.jacobian(Pi_a, _efield, experimental_use_pfor=False)
            epsilon_infty *= UNIT_FACTOR #
            epsilon_infty += tf.eye(3)
        else:
            epsilon_infty = tf.eye(3)
            Zstar = tf.zeros((nat,3,3))

        #differentiating a scalar w.r.t tensors
        forces = tape0.gradient(total_energy, positions)
        #needs tape to be persistent
        dE_dh = tape0.gradient(total_energy, cell)
        #V = tf.abs(tf.tensordot(cell[0], tf.linalg.cross(cell[1], cell[2]),axes=1))
        # Stress: stress_{ij} = (1/V) \sum_k dE/dh_{ik} * h_{jk}
        stress = tf.linalg.matmul(dE_dh, cell, transpose_b=True) / cell_volume
        pad_rows = tf.zeros([nmax_diff, tf.shape(forces)[1]], dtype=forces.dtype)
        forces = tf.concat([-forces, pad_rows], axis=0)

        pad_rows = tf.zeros([nmax_diff, 3, 3], dtype=tf.float32)
        Zstar = tf.concat([Zstar, pad_rows], axis=0)

        pad_rows = tf.zeros([nmax_diff], dtype=tf.float32)
        charges = tf.concat([charges, pad_rows], axis=0)
        E1 = tf.concat([E1, pad_rows], axis=0)
        E2 = tf.concat([E2, pad_rows], axis=0)
        C6 = tf.zeros(nat+nmax_diff)
        Pi_a = tf.zeros(3)

        return [total_energy, forces, C6, charges, stress, 
                Pi_a, E1, E2, Zstar, epsilon_infty]
    
    @tf.function(jit_compile=False,
                input_signature=[(
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                )])
    def tf_predict_energy_forces_pqeq0(self, x):

        '''
        x = (batch_species_encoder,positions,
                    nmax_diff, batch_nats,cells,C6,
                first_atom_idx,second_atom_idx,shift_vectors,num_pairs)
        '''
        rc = tf.constant(self.rcut,dtype=tf.float32)
        Nrad = tf.constant(self.Nrad, dtype=tf.int32)
        #thetasN = tf.constant(self.thetaN, dtype=tf.int32)

        nat = x[3]
        nmax_diff = x[2]

        species_encoder = tf.reshape(x[0][:nat*self.nspec_embedding],
                                     [nat,self.nspec_embedding])

        #width = tf.cast(x[1], dtype=tf.float32)
        positions = tf.reshape(x[1][:nat*3], [nat,3])
        positions = positions
        cell = tf.reshape(x[4], [3,3])
        evdw = 0.0
        C6 = x[5][:nat]
        num_pairs = x[9]
        first_atom_idx = tf.cast(x[6][:num_pairs], tf.int32)
        second_atom_idx = tf.cast(x[7][:num_pairs],tf.int32)
        shift_vector = tf.cast(tf.reshape(x[8][:num_pairs*3],
                                          [num_pairs,3]), tf.float32)
        gaussian_width = tf.reshape(x[10][:nat*2], [nat,2])
        chi0 = x[11][:nat]
        J0 = x[12][:nat]
        atomic_q0 = x[13][:nat]
        total_charge = x[14]
        nuclei_charge = x[15][:nat]
        atomic_number = x[16][:nat]
        _efield = self.efield
        cell_volume = tf.abs(tf.linalg.det(cell))
        with tf.GradientTape(persistent=True) as tape0:
            tape0.watch(positions)
            tape0.watch(cell)
            tape0.watch(_efield)

            #based on ase 
            #npairs x 3
            all_rij = tf.gather(positions,second_atom_idx) - \
                     tf.gather(positions,first_atom_idx) + \
                    tf.tensordot(shift_vector,cell,axes=1)

            all_rij_norm = tf.linalg.norm(all_rij, axis=-1) #npair
            reg = 1e-12
            #all_rij_norm = tf.sqrt(tf.reduce_sum(all_rij * all_rij , axis=-1) + reg) #npair
            species_encoder_i = tf.gather(species_encoder,first_atom_idx)
            species_encoder_j = tf.gather(species_encoder,second_atom_idx)

            species_encoder_extended = species_encoder_i * species_encoder_j

            species_encoder_ij = tf.reshape(species_encoder_extended, 
                                                  [-1, self.spec_size]
                                                   ) #npairs,spec_size

            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x Nneigh x embedding
            #fcuts is nat x Nneigh and reshaped to nat x Nneigh
            #_Nneigh = tf.shape(all_rij_norm)
            #neigh = _Nneigh[1]
            kn_rad = tf.ones(self.n_bessels,dtype=tf.float32)
            bf_radial0 = help_fn.bessel_function(all_rij_norm,
                                            rc,kn_rad,
                                            self.n_bessels) #
            

            bf_radial = tf.reshape(bf_radial0, [-1,self.n_bessels])
            bf_radial = self.radial_funct_net(bf_radial)
            bf_radial = tf.reshape(bf_radial, [num_pairs, self.Nrad, self.number_radial_components])
            radial_ij = tf.expand_dims(bf_radial, 2) * tf.expand_dims(tf.expand_dims(species_encoder_ij, 1), -1)
            radial_ij = tf.reshape(radial_ij, [num_pairs, self.Nrad*self.spec_size,self.number_radial_components])
            atomic_descriptors = tf.math.unsorted_segment_sum(data=radial_ij[:,:,0],
                                                              segment_ids=first_atom_idx, num_segments=nat) 

            #implement angular part: compute vect_rij dot vect_rik / rij / rik
            #rij_unit = tf.einsum('ij,i->ij',all_rij, 1.0 / (all_rij_norm+reg)) #npair,3
            rij_unit = all_rij / (tf.expand_dims(all_rij_norm + reg, -1))

            Gi = self.to_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat)
            Gi = tf.concat(Gi, axis=1)
            atomic_descriptors = tf.concat([atomic_descriptors, Gi], axis=1)
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])
            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            total_energy = tf.reduce_sum(_atomic_energies[:,0])
            #########################
            idx = 1
            E1 = tf.nn.softplus(_atomic_energies[:,idx])
            idx += 1
            E2 = tf.nn.softplus(_atomic_energies[:,idx])
            idx += 1
            E_d2 = 1.0 + tf.tile(tf.nn.softplus(_atomic_energies[:,idx])[:,None], [1,3]) # eV/A^2 [1,infty]
            #############################################
            #include atomic electronegativity(chi0) and hardness (J0)
            E1 += chi0
            E2 += J0

            _b = tf.identity(E1)
            _b -= E2 * atomic_q0 # only if we are not optimizing deq

            _ewald = ewald(positions, cell, nat,
                    gaussian_width,self.accuracy,
                           None, self.pbc, _efield,
                           self.gaussian_width_scale
                           )

            if self.apply_field:
                #directly diffrentiate charge and shell_disp.
                #Since the derivative is independent on charge and shell_disp, we are initializing them to zero
                #Not: this doesn's affect the derivatives needed here
                shell_disp = tf.zeros((nat,3))[:,None,:]
                charges = atomic_q0
                #field_kernel_q, field_kernel_e, field_kernel_qe = _ewald.atom_centered_dV_2(shell_disp,
                field_kernel_q, field_kernel_e = atom_centered_dV(all_rij,shell_disp,
                                                                  nuclei_charge,charges,
                                                                  first_atom_idx,second_atom_idx,
                                                                  atomic_number, self.central_atom_id,
                                                                  _efield)
                field_kernel_qe = tf.zeros((nat,self._n_shells*3))
            else:
                field_kernel_q,field_kernel_e, field_kernel_qe = tf.zeros(nat), tf.zeros((nat,3)), tf.zeros((nat,3))

            _b += field_kernel_q
            if self.exact_solver:
                Vij, Vija, Vijab = _ewald.recip_space_term_with_shelld_quadratic_qd(nuclei_charge)
                charges, shell_disp = _compute_charges_disp(Vij, Vija,Vijab,
                     _b, E2, E_d2, field_kernel_e,
                     atomic_q0, total_charge)
                ecoul = _compute_coulumb_energy_pqeq_qd(charges, atomic_q0,
                            E1, E2, shell_disp, Vij, Vija, Vijab)
                ecoul += 0.5 * tf.reduce_sum(E_d2 * shell_disp * shell_disp)
            else:

                #solve linear system using preconditioned conjugate gradients
                @tf.function
                def A_matvec(v):
                    v = tf.reshape(v, [-1])
                    return tf.reshape(_ewald.A_matvec_pqeq0(v, E2, E_d2), [-1,1])
                    #return _ewald.A_matvec_pqeq0(v, E2, E_d2)
                #@tf.function
                #def M_inv():
                    #v = tf.reshape(v, [-1])
                    #return tf.reshape(_ewald.M_inv_pqeq0(v, E2, E_d2), [-1,1])
                #    return tf.reshape(_ewald.M_inv_pqeq0(v, E2, E_d2), [-1,1])
                    #return _ewald.M_inv_pqeq0(v, E2, E_d2)

                b = tf.concat([-_b, [total_charge],-tf.reshape(field_kernel_e, [-1])], axis=0)
                x0 = tf.concat([atomic_q0, [0.0], tf.zeros(nat*3*self._n_shells)], axis=0)
                #x0 = tf.zeros(4*nat+1)
                n = b.shape[0]
                linop = AOperator(A_matvec, n)
                linop_M_inv = tf.linalg.LinearOperatorDiag(
                        _ewald.M_inv_pqeq0(E2, E_d2))
                results = tf.linalg.experimental.conjugate_gradient(
                            linop,
                            b,
                            x=x0,
                            preconditioner=linop_M_inv,
                            tol=1e-6,
                            max_iter=100
                        )
                #this is to prevect backpropagation through the cg loop. 
                #Very important to get the correct forces. 
                #This is correct because the forces do not depend on the derivative of q or p 
                # w.r.t atomic position, thanks to pqeq
                charges_disp = tf.stop_gradient(results[1])

                charges = charges_disp[:nat]
                shell_disp = tf.reshape(charges_disp[nat+1:], [nat,3])

                ecoul = _ewald.coulumb_energy_qd(charges_disp)
                dq = charges - atomic_q0
                ecoul += tf.reduce_sum(E1 * dq + 0.5 * E2 * dq * dq)
                ecoul += 0.5 * tf.reduce_sum(E_d2 * shell_disp * shell_disp)

            #if self.apply_field:
            _shell_disp = tf.reshape(shell_disp, [nat,self._n_shells,3])
            Piq_a, Pie_a = atom_centered_polarization(all_rij,_shell_disp,
                                                          nuclei_charge,charges,
                                                          first_atom_idx,second_atom_idx,
                                                          atomic_number, self.central_atom_id)
            #else:
            #    Piq_a, Pie_a = tf.zeros(3), tf.zeros((self._n_shells,3))

            Pi_a = Piq_a + tf.reduce_sum(Pie_a, axis=0)
            efield_energy = -(tf.reduce_sum(Piq_a * _efield) +
                              tf.reduce_sum(Pie_a * _efield[None,:]))

            ecoul += efield_energy
            total_energy += ecoul
            
            Pi_a /= cell_volume

        #differentiating a scalar w.r.t tensors
        if self.apply_field:
            #Z*_{iab} = V * dP_a/dr_{ib}
            Zstar = cell_volume * tf.transpose(tape0.jacobian(Pi_a, positions, 
                                                              experimental_use_pfor=False), perm=(1,0,2))
            #epsilon*_{ab} = dP_a/defield_b
            epsilon_infty = tape0.jacobian(Pi_a, _efield, experimental_use_pfor=False)
            epsilon_infty *= UNIT_FACTOR #
            epsilon_infty += tf.eye(3)
        else:
            epsilon_infty = tf.eye(3)
            Zstar = tf.zeros((nat,3,3))
        forces = tape0.gradient(total_energy, positions)
        #needs tape to be persistent
        dE_dh = tape0.gradient(total_energy, cell)
        # Stress: stress_{ij} = (1/V) \sum_k dE/dh_{ik} * h_{jk}
        stress = tf.linalg.matmul(dE_dh, cell, transpose_b=True) / cell_volume
        pad_rows = tf.zeros([nmax_diff, tf.shape(forces)[1]], dtype=forces.dtype)
        forces = tf.concat([-forces, pad_rows], axis=0)
        shell_disp = tf.concat([shell_disp, pad_rows], axis=0)
        E_d2 = tf.concat([E_d2, pad_rows], axis=0)

        pad_rows = tf.zeros([nmax_diff, 3, 3], dtype=tf.float32)
        Zstar = tf.concat([Zstar, pad_rows], axis=0)

        pad_rows = tf.zeros([nmax_diff], dtype=tf.float32)
        charges = tf.concat([charges, pad_rows], axis=0)
        E1 = tf.concat([E1, pad_rows], axis=0)
        E2 = tf.concat([E2, pad_rows], axis=0)

        return [total_energy, forces, C6, charges, stress, shell_disp, 
                Pi_a, E1, E2, E_d2, Zstar, epsilon_infty]
    @tf.function(jit_compile=False,
                input_signature=[(
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                )])
    def tf_predict_energy_forces_pqeq0_n(self, x):

        '''
        x = (batch_species_encoder,positions,
                    nmax_diff, batch_nats,cells,C6,
                first_atom_idx,second_atom_idx,shift_vectors,num_pairs)
        '''
        rc = tf.constant(self.rcut,dtype=tf.float32)
        Nrad = tf.constant(self.Nrad, dtype=tf.int32)
        #thetasN = tf.constant(self.thetaN, dtype=tf.int32)

        nat = x[3]
        nmax_diff = x[2]

        species_encoder = tf.reshape(x[0][:nat*self.nspec_embedding],
                                     [nat,self.nspec_embedding])

        #width = tf.cast(x[1], dtype=tf.float32)
        positions = tf.reshape(x[1][:nat*3], [nat,3])
        positions = positions
        cell = tf.reshape(x[4], [3,3])
        evdw = 0.0
        C6 = x[5][:nat]
        num_pairs = x[9]
        first_atom_idx = tf.cast(x[6][:num_pairs], tf.int32)
        second_atom_idx = tf.cast(x[7][:num_pairs],tf.int32)
        shift_vector = tf.cast(tf.reshape(x[8][:num_pairs*3],
                                          [num_pairs,3]), tf.float32)
        gaussian_width = tf.reshape(x[10][:nat*2], [nat,2])
        chi0 = x[11][:nat]
        J0 = x[12][:nat]
        atomic_q0 = x[13][:nat]
        total_charge = x[14]
        nuclei_charge = x[15][:nat]
        atomic_number = x[16][:nat]
        _efield = self.efield
        with tf.GradientTape(persistent=True) as tape0:
            tape0.watch(positions)
            tape0.watch(cell)
            tape0.watch(_efield)

            #based on ase 
            #npairs x 3
            all_rij = tf.gather(positions,second_atom_idx) - \
                     tf.gather(positions,first_atom_idx) + \
                    tf.tensordot(shift_vector,cell,axes=1)

            all_rij_norm = tf.linalg.norm(all_rij, axis=-1) #npair
            reg = 1e-12
            #all_rij_norm = tf.sqrt(tf.reduce_sum(all_rij * all_rij , axis=-1) + reg) #npair
            species_encoder_i = tf.gather(species_encoder,first_atom_idx)
            species_encoder_j = tf.gather(species_encoder,second_atom_idx)

            species_encoder_extended = species_encoder_i * species_encoder_j

            species_encoder_ij = tf.reshape(species_encoder_extended, 
                                                  [-1, self.spec_size]
                                                   ) #npairs,spec_size

            #since fcut =0 for rij > rc, there is no need for any special treatment
            #species_encoder Nneigh and reshaped to nat x Nneigh x embedding
            #fcuts is nat x Nneigh and reshaped to nat x Nneigh
            #_Nneigh = tf.shape(all_rij_norm)
            #neigh = _Nneigh[1]
            kn_rad = tf.ones(self.n_bessels,dtype=tf.float32)
            bf_radial0 = help_fn.bessel_function(all_rij_norm,
                                            rc,kn_rad,
                                            self.n_bessels) #

            bf_radial1 = tf.reshape(bf_radial0, [-1,self.n_bessels])
            bf_radial2 = self.radial_funct_net(bf_radial1)
            bf_radial = tf.reshape(bf_radial2, [num_pairs, self.Nrad, self.number_radial_components])
            radial_ij = tf.expand_dims(bf_radial, 2) * tf.expand_dims(tf.expand_dims(species_encoder_ij, 1), -1)
            radial_ij = tf.reshape(radial_ij, [num_pairs, self.Nrad*self.spec_size,self.number_radial_components])
            atomic_descriptors = tf.math.unsorted_segment_sum(data=radial_ij[:,:,0],
                                                              segment_ids=first_atom_idx, num_segments=nat) 

            #implement angular part: compute vect_rij dot vect_rik / rij / rik
            #rij_unit = tf.einsum('ij,i->ij',all_rij, 1.0 / (all_rij_norm+reg)) #npair,3
            rij_unit = all_rij / (tf.expand_dims(all_rij_norm + reg, -1))

            Gi = self.to_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat)

            #for nbody in tf.range(0, self.body_order-3): # start from 3-body
            #    #idx = nbody - 3
            #    atomic_descriptors = tf.concat([atomic_descriptors, Gi[nbody]], axis=1)
            Gi = tf.concat(Gi, axis=1)
            atomic_descriptors = tf.concat([atomic_descriptors, Gi], axis=1)
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])

            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            total_energy = tf.reduce_sum(_atomic_energies[:,0])
            #########################
            idx = 1
            E1 = tf.nn.softplus(_atomic_energies[:,idx])
            idx += 1
            E2 = tf.nn.softplus(_atomic_energies[:,idx])
            idx += 1
            E_d2 = tf.tile(tf.nn.softplus(_atomic_energies[:,idx:])[...,None], [1,1,3]) # eV/A^2 [1,infty]

            #E_d2 = tf.tile(
            #        tf.clip_by_value(tf.nn.softplus(_atomic_energies[:,idx:]), 
            #                         clip_value_min=0.1, clip_value_max=5.0)[...,None], [1,1,3]) # eV/A^2 [1,infty]
            E_d2 = tf.reshape(E_d2, [nat,3*self._n_shells])

            #############################################
            #include atomic electronegativity(chi0) and hardness (J0)
            E1 += chi0
            E2 += J0

            _b = tf.identity(E1)
            _b -= E2 * atomic_q0 # only if we are not optimizing deq

            _ewald = ewald(positions, cell, nat,
                    gaussian_width,self.accuracy,
                           None, self.pbc, _efield,
                           self.gaussian_width_scale
                           )

            if self.apply_field:
                 #directly diffrentiate charge and shell_disp.
                #Since the derivative is independent on charge and shell_disp, we are initializing them to zero
                #Not: this doesn's affect the derivatives needed here
                shell_disp = tf.zeros((nat, self._n_shells,3))
                charges = tf.identity(atomic_q0)
                field_kernel_q, field_kernel_e, field_kernel_qe = _ewald.atom_centered_dV_2(shell_disp,
                                    charges, nuclei_charge,
                                    self.central_atom_id,
                                    atomic_number,
                                    n_shells=self._n_shells)
                _b += field_kernel_q # the term coming from qi-nuclei. The electronic contribution does not contribute to change in nuclei charges
            else:
                field_kernel_q,field_kernel_e, field_kernel_qe = tf.zeros(nat), tf.zeros((nat,3*self._n_shells)), tf.zeros((nat,3*self._n_shells))

            _b += field_kernel_q

            Vij, Vij_qz, Vij_zq, Vij_zz = _ewald.recip_space_term_with_shelld_quadratic_q_nd(self._n_shells,nuclei_charge)
            charges, shell_disp = _compute_charges_disp(Vij, Vij_qz, Vij_zq, Vij_zz,_b, E2, E_d2, 
                                                        field_kernel_e, field_kernel_qe, atomic_q0, total_charge)
            Vij_qz = tf.reshape(Vij_qz, [nat,nat,-1])
            Vij_zq = tf.reshape(Vij_zq, [nat,nat,-1])
            Vij_zz = tf.reshape(tf.transpose(Vij_zz, [0,1,2,4,3,5]), [nat,nat,3*self._n_shells,3*self._n_shells])
            ecoul = _compute_coulumb_energy_pqeq_qd(charges, atomic_q0,
                        E1, E2, shell_disp, Vij, Vij_qz, Vij_zq, Vij_zz)
            dq = charges - atomic_q0

            ecoul += 0.5 * tf.reduce_sum(E_d2 * shell_disp * shell_disp) 
            if self.apply_field:
                _shell_disp = tf.reshape(shell_disp, [nat, self._n_shells, 3])
                Piq_a, Pie_a = _ewald.atom_centered_polarization_2(_shell_disp, positions,
                                                             nuclei_charge,
                                                             charges,
                                                             self.central_atom_id,
                                                             atomic_number,
                                                             n_shells=self._n_shells)
            else:
                Piq_a, Pie_a = tf.zeros(3), tf.zeros((self._n_shells,3))

            Pi_a = Piq_a + tf.reduce_sum(Pie_a, axis=0)
            efield_energy = -(tf.reduce_sum(Piq_a * _efield) + 
                              tf.reduce_sum(Pie_a * _efield[None,:]))

            ecoul += efield_energy
            total_energy += ecoul
            
            cell_volume = tf.abs(tf.linalg.det(cell))
            Pi_a /= cell_volume

        #differentiating a scalar w.r.t tensors
        if self.apply_field:
            #Z*_{iab} = V * dP_a/dr_{ib}
            Zstar = cell_volume * tf.transpose(tape0.jacobian(Pi_a, positions, experimental_use_pfor=False), perm=(1,0,2))
            #epsilon*_{ab} = dP_a/defield_b
            epsilon_infty = tape0.jacobian(Pi_a, _efield, experimental_use_pfor=False)
            epsilon_infty *= UNIT_FACTOR #
            epsilon_infty += tf.eye(3)
        else:
            epsilon_infty = tf.eye(3)
            Zstar = tf.zeros((nat,3,3))
        forces = tape0.gradient(total_energy, positions)
        #needs tape to be persistent
        dE_dh = tape0.gradient(total_energy, cell)
        # Stress: stress_{ij} = (1/V) \sum_k dE/dh_{ik} * h_{jk}
        stress = tf.linalg.matmul(dE_dh, cell, transpose_b=True) / cell_volume
        pad_rows = tf.zeros([nmax_diff, tf.shape(forces)[1]], dtype=forces.dtype)
        forces = tf.concat([-forces, pad_rows], axis=0)
        pad_rows = tf.zeros([nmax_diff, 3*self._n_shells], dtype=forces.dtype)
        shell_disp = tf.concat([shell_disp, pad_rows], axis=0)
        E_d2 = tf.concat([E_d2, pad_rows], axis=0)

        pad_rows = tf.zeros([nmax_diff, 3, 3], dtype=tf.float32)
        Zstar = tf.concat([Zstar, pad_rows], axis=0)

        pad_rows = tf.zeros([nmax_diff], dtype=tf.float32)
        charges = tf.concat([charges, pad_rows], axis=0)
        E1 = tf.concat([E1, pad_rows], axis=0)
        E2 = tf.concat([E2, pad_rows], axis=0)

        return [total_energy, forces, C6, charges, stress, shell_disp, 
                Pi_a, E1, E2, E_d2, Zstar, epsilon_infty]
