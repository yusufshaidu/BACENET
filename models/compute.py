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
from models.descriptors import (to_three_body_order_terms, to_four_body_order_terms,
                                to_five_body_order_terms)

from models.coulomb_functions import (_compute_Aij, _compute_Fia, _compute_Fiajb, 
                                      _compute_charges_disp, _compute_shell_disp_qqdd2, 
                                      _compute_shell_disp_qqdd1,_compute_charges,
                                      _compute_coulumb_energy,
                                      _compute_coulumb_energy_pqeq_qd,
                                      _compute_coulumb_energy_pqeq, run_scf)
from data.unpack_tfr_data import unpack_data
from models.ewald import ewald

import warnings
import logging
constant_e = 1.602176634e-19

#tf.config.run_functions_eagerly(
#    True
#)

#tf.debugging.enable_check_numerics()
class Compute:
    def __init__(self, configs):

        #network section
        self.rcut = configs['rc_rad']
        self.Nrad = int(configs['Nrad'])
        self.zeta = configs['zeta'] # this is a list of l max per body order
        self.body_order = configs['body_order']
        self.species_correlation = configs['species_correlation']
        self.species = configs['species']
        self.nspecies = len(self.species)
        self.species_layer_sizes = configs['species_layer_sizes']

        self.nspec_embedding = self.species_layer_sizes[-1]
        self.spec_size = configs['spec_size']
        self.feature_size = configs['feature_size']
        
        self.n_bessels = configs['n_bessels'] if not None else self.Nrad
        self.n_bessels = int(self.n_bessels)

        #dispersion parameters and electrostatics
        self.include_vdw = configs['include_vdw']
        self.rmin_u = configs['rmin_u']
        self.rmax_u = configs['rmax_u']
        self.rmin_d = configs['rmin_d']
        self.rmax_d = configs['rmax_d']
        # the number of elements in the periodic table
        self.nelement = configs['nelement']

        self.efield = configs['efield']
        if self.efield is not None:
            self.efield = tf.cast(self.efield, tf.float32)
            self.apply_field = True
        else:
            self.efield = tf.cast([0.0,0.0,0.0], tf.float32)
            self.apply_field = False
        print('field', self.efield)
        
        self.accuracy = configs['accuracy']
        self.pbc = configs['pbc']
        self.central_atom_id = configs['central_atom_id']

        self.gaussian_width_scale = configs['gaussian_width_scale']
        
        self.zeta = configs['zeta']

        self.number_radial_components = 1
        for i, z in enumerate(self.zeta):
            self.number_radial_components += (i+1) * z + 1 # 1 for 2body, l_3 + 1 for 3body, 2 l_4 + 1 for 4body, 3 l_5 + 1 for 5 body

        self.atomic_nets = configs['atomic_nets']
        self.radial_funct_net = configs['radial_funct_net']


        if configs['body_order'] == 3:
            self.to_body_order_terms = to_three_body_order_terms
        elif configs['body_order'] == 4:
            self.to_body_order_terms = to_four_body_order_terms
        elif configs['body_order'] == 5:
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
            #for nbody in tf.range(0, self.body_order-3):
            #    atomic_descriptors = tf.concat([atomic_descriptors, Gi[nbody]], axis=1)

            #field components = \sum_j{fc(rij) * (E . rij_unit)}
            #_efield_extended = tf.squeeze(tf.matmul(rij_unit, _efield[:, None])) * help_fn.tf_fcut_rbf(all_rij_norm, rc)
            #_efield_extended = tf.math.unsorted_segment_sum(data=_efield_extended,
            #                                                  segment_ids=first_atom_idx, num_segments=nat)

            #atomic_descriptors = tf.concat([atomic_descriptors, _efield_extended[:,None]], axis=1) # append electric field
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



            field_kernel_q, field_kernel_e, field_kernel_qe = tf.cond(
                tf.constant(self.apply_field),
                lambda: _ewald.atom_centered_dV(
                    nuclei_charge,
                    self.central_atom_id,
                    atomic_number,
                ),
                lambda: (
                    tf.zeros_like(nuclei_charge),          # shape (nat,)
                    tf.zeros((nat, 3), dtype=tf.float32),  # shape (nat,3)
                    tf.zeros((nat, 3), dtype=tf.float32),  # shape (nat,3)
                )
                )

            _b += field_kernel_q

            charges = self.compute_charges(Vij, _b, E2, atomic_q0, total_charge)
            ecoul = _compute_coulumb_energy(charges, atomic_q0, E1, E2, Vij)

            Piq_a, Pie_a = tf.cond(tf.constant(self.apply_field),
                                   lambda: _ewald.atom_centered_polarization(tf.zeros((nat,3)),
                                                             nuclei_charge,
                                                             charges,
                                                             self.central_atom_id,
                                                             atomic_number
                                                             ),
                                   lambda: (
                                            tf.zeros(3),          # shape (3,)
                                            tf.zeros(3),          # shape (3,)
                                            )
                                   )
            Pi_a = Piq_a + Pie_a
            efield_energy = -tf.reduce_sum((Piq_a + Pie_a) * _efield)
            ecoul += efield_energy 
            Vj = _ewald.recip_space_term_with_shelld_linear_Vj(shell_disp,
                                               nuclei_charge,
                                               charges)
            total_energy += ecoul

        #differentiating a scalar w.r.t tensors
        pad_rows = tf.zeros([nmax_diff], dtype=tf.float32)
        charges = tf.concat([charges, pad_rows], axis=0)
        Pi_a = -tape0.gradient(total_energy, _efield) #contains P0 - dE/d_eps
        #total_energy -= tf.reduce_sum(Pi_a * _efield)
        forces = tape0.gradient(total_energy, positions)
        #needs tape to be persistent
        dE_dh = tape0.gradient(total_energy, cell)
        V = tf.abs(tf.tensordot(cell[0], tf.linalg.cross(cell[1], cell[2]),axes=1))
            # Stress: stress_{ij} = (1/V) \sum_k dE/dh_{ik} * h_{jk}
        stress = tf.linalg.matmul(dE_dh, cell, transpose_b=True) / V
        pad_rows = tf.zeros([nmax_diff, tf.shape(forces)[1]], dtype=forces.dtype)
        forces = tf.concat([-forces, pad_rows], axis=0)
        shell_disp = tf.concat([shell_disp, pad_rows], axis=0)
        return [total_energy, forces, C6, charges, stress, Pi_a, E1, E2, Vj]

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

        with tf.GradientTape(persistent=True) as tape0:
            tape0.watch(positions)
            tape0.watch(cell)
            #tape0.watch(_efield)

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

            #field components = \sum_j{fc(rij) * (E . rij_unit)}
            #_efield_extended = tf.squeeze(tf.matmul(rij_unit, _efield[:, None])) * help_fn.tf_fcut_rbf(all_rij_norm, rc)
            #_efield_extended = tf.math.unsorted_segment_sum(data=_efield_extended,
            #                                                  segment_ids=first_atom_idx, num_segments=nat)

            #atomic_descriptors = tf.concat([atomic_descriptors, _efield_extended[:,None]], axis=1) # append electric field
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])

            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            total_energy = tf.reduce_sum(_atomic_energies[:,0])
            idx = 1
            E1 = tf.nn.softplus(_atomic_energies[:,idx])
            idx += 1
            E2 = tf.nn.softplus(_atomic_energies[:,idx])
            #E_d1 = tf.tile(tf.math.tanh(_atomic_energies[:,idx])[:,None], [1,3]) * 0.1 # eV/A [-0.1,0.1]
            E_d1 = tf.zeros_like(positions)
            idx += 1
            E_d2 = 1.0 + tf.tile(tf.nn.softplus(_atomic_energies[:,idx])[:,None], [1,3]) # eV/A^2 [2,infty]
            #idx += 1
            #E_qd = tf.tile(tf.math.tanh(_atomic_energies[:,idx])[:,None], [1,3]) * 0.1 # eV/A [-0.1,0.1]
            E_qd = tf.zeros_like(positions)

            #include atomic electronegativity(chi0) and hardness (J0)
            E1 += chi0
            E2 += J0

            _b = tf.identity(E1)
            _b -= E2 * atomic_q0 # only if we are not optimizing deq

            _ewald = ewald(positions, cell, nat,
                    gaussian_width,self.accuracy,
                           None, self.pbc, self.efield,
                           self.gaussian_width_scale
                           )

            if self.apply_field:
                field_kernel_q, field_kernel_e, field_kernel_qe = _ewald.atom_centered_dV(nuclei_charge,
                                                                                     self.central_atom_id,
                                                                                     atomic_number)
                _b += field_kernel_q # the term coming from qi-nuclei. The electronic contribution does not contribute to change in nuclei charges
            else:
                field_kernel_q,field_kernel_e, field_kernel_qe = tf.zeros(nat), tf.zeros((nat,3)), tf.zeros((nat,3))

            _b += field_kernel_q
            #E_d1 += field_kernel_e # the linear term in d
            _E_d1 = (E_d1 + field_kernel_e - 0.5 * E_qd * atomic_q0[:,None])
            Vij, Vij_qz, Vij_zq, Vij_zz = _ewald.recip_space_term_with_shelld_quadratic_qd(nuclei_charge)
            #E_d1 = field_kernel_e
            charges, shell_disp = _compute_charges_disp(Vij, Vij_qz, Vij_zq, Vij_zz,
                 _b, E2, E_d2, _E_d1, E_qd + field_kernel_qe, # field_kernel_qe comes from placing Z at the nuclei position and q-Z at the shell
                 atomic_q0, total_charge)
            ecoul = _compute_coulumb_energy_pqeq_qd(charges, atomic_q0,
                        E1, E2, shell_disp, Vij, Vij_qz, Vij_zq, Vij_zz)
            dq = charges - atomic_q0
            ecoul += tf.reduce_sum((E_d1  + 0.5 * E_qd * dq[:,None] + 0.5 *  E_d2 * shell_disp) * shell_disp) 
            if self.apply_field:
                Piq_a, Pie_a = _ewald.atom_centered_polarization(shell_disp,
                                                             nuclei_charge,
                                                             charges,
                                                             self.central_atom_id,
                                                             atomic_number
                                                             )
            else:
                Piq_a, Pie_a = tf.zeros(3), tf.zeros(3)

            Pi_a = Piq_a + Pie_a
            efield_energy = -tf.reduce_sum(Pi_a * self.efield)

            ecoul += efield_energy
            Vj = _ewald.recip_space_term_with_shelld_linear_Vj(shell_disp,
                                               nuclei_charge,
                                               charges)
            total_energy += ecoul

        #differentiating a scalar w.r.t tensors
        pad_rows = tf.zeros([nmax_diff], dtype=tf.float32)
        charges = tf.concat([charges, pad_rows], axis=0)
        #Pi_a = -tape0.gradient(total_energy, _efield) #contains P0 - dE/d_eps
        #total_energy -= tf.reduce_sum(Pi_a * _efield)
        forces = tape0.gradient(total_energy, positions)
        #needs tape to be persistent
        dE_dh = tape0.gradient(total_energy, cell)
        V = tf.abs(tf.tensordot(cell[0], tf.linalg.cross(cell[1], cell[2]),axes=1))
        # Stress: stress_{ij} = (1/V) \sum_k dE/dh_{ik} * h_{jk}
        stress = tf.linalg.matmul(dE_dh, cell, transpose_b=True) / V
        pad_rows = tf.zeros([nmax_diff, tf.shape(forces)[1]], dtype=forces.dtype)
        forces = tf.concat([-forces, pad_rows], axis=0)
        shell_disp = tf.concat([shell_disp, pad_rows], axis=0)
        return [total_energy, forces, C6, charges, stress, shell_disp, Pi_a, E1, E2, E_d2,E_d1,E_qd, Vj]

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
    def tf_predict_energy_forces_pqeq1(self, x):
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

        with tf.GradientTape(persistent=True) as tape0:
            tape0.watch(positions)
            tape0.watch(cell)
            #tape0.watch(_efield)

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
            #for nbody in tf.range(0, self.body_order-3):
            #    atomic_descriptors = tf.concat([atomic_descriptors, Gi[nbody]], axis=1)
            Gi = tf.concat(Gi, axis=1)
            atomic_descriptors = tf.concat([atomic_descriptors, Gi], axis=1)

            #field components = \sum_j{fc(rij) * (E . rij_unit)}
            #_efield_extended = tf.squeeze(tf.matmul(rij_unit, _efield[:, None])) * help_fn.tf_fcut_rbf(all_rij_norm, rc)
            #_efield_extended = tf.math.unsorted_segment_sum(data=_efield_extended,
            #                                                  segment_ids=first_atom_idx, num_segments=nat)

            #atomic_descriptors = tf.concat([atomic_descriptors, _efield_extended[:,None]], axis=1) # append electric field
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])

            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            total_energy = tf.reduce_sum(_atomic_energies[:,0])
            idx = 1
            E1 = tf.nn.softplus(_atomic_energies[:,idx])
            idx += 1
            E2 = tf.nn.softplus(_atomic_energies[:,idx])
            #E_d1 = tf.tile(tf.math.tanh(_atomic_energies[:,idx])[:,None], [1,3]) * 0.1 # eV/A [-0.1,0.1]
            E_d1 = tf.zeros_like(positions)
            idx += 1
            E_d2 = 1.0 + tf.tile(tf.nn.softplus(_atomic_energies[:,idx])[:,None], [1,3]) # eV/A^2 [2,infty]
            #idx += 1
            #E_qd = tf.tile(tf.math.tanh(_atomic_energies[:,idx])[:,None], [1,3]) * 0.1 # eV/A [-0.1,0.1]
            E_qd = tf.zeros_like(positions)

            #include atomic electronegativity(chi0) and hardness (J0)
            E1 += chi0
            E2 += J0

            _b = tf.identity(E1)
            _b -= E2 * atomic_q0 # only if we are not optimizing deq

            _ewald = ewald(positions, cell, nat,
                    gaussian_width,self.accuracy,
                           None, self.pbc, self.efield,
                           self.gaussian_width_scale
                           )
            if self.apply_field:
                field_kernel_q, field_kernel_e, field_kernel_qe = _ewald.atom_centered_dV(nuclei_charge,
                                                                                     self.central_atom_id,
                                                                                     atomic_number)
                _b += field_kernel_q # the term coming from qi-nuclei. The electronic contribution does not contribute to change in nuclei charges
            else:
                field_kernel_q,field_kernel_e, field_kernel_qe = tf.zeros(nat), tf.zeros((nat,3)), tf.zeros((nat,3))

            _b += field_kernel_q

            '''
            Vij, Vij_qz, Vij_zq, Vij_zz, Vij_qz2, Vij_zq2 = _ewald.recip_space_term_with_shelld_quadratic_qqdd_1(nuclei_charge)
            # compute charges at d = 0
            charges, shell_disp = run_scf(Vij, Vij_qz, Vij_zq, Vij_zz, Vij_qz2, Vij_zq2,
                    E_d1, E_d2, E_qd, E2,
                    atomic_q0, total_charge,
                    field_kernel_qe, field_kernel_e, _b,
                    tol=1e-3, max_iter=4)
            '''
            Vij, Vij_qz, Vij_zq, Vij_zz, Vij_qz2, Vij_zq2 = _ewald.recip_space_term_with_shelld_quadratic_qqdd_1(nuclei_charge)
            # compute charges at d = 0
            charges = _compute_charges(Vij, _b, E2, atomic_q0, total_charge)
            # determine d
            shell_disp = _compute_shell_disp_qqdd1(Vij_qz, Vij_zq, Vij_qz2, Vij_zq2, Vij_zz, E_d1,
               E_d2 + field_kernel_qe, E_qd, atomic_q0,charges, field_kernel_e)

            #update charges
            shell_d2 = shell_disp[:,:,None] * shell_disp[:,None,:]
            #_b += 0.5 * tf.reduce_sum((tf.transpose(Vij_zq,perm=(1,0,2)) + Vij_qz) * shell_disp[None,:,:], axis=(1,2)) #N
            #_b += 0.5 * tf.reduce_sum((tf.transpose(Vij_zq2,perm=(1,0,2,3)) + Vij_qz2) * shell_d2[None,...], axis=(1,2,3)) #N
            #_b += 0.5 * tf.reduce_sum(E_qd * shell_disp, axis=1) #sum over cartessian directions
            #charges = _compute_charges(Vij, _b, E2, atomic_q0, total_charge)
            #'''

            dq = charges - atomic_q0

            ecoul = _compute_coulumb_energy_pqeq_qd(charges, atomic_q0,
                        E1, E2, shell_disp, Vij, Vij_qz, Vij_zq, Vij_zz)
            #add additional terms to energy
            #shell_d2 = shell_disp[:,:,None] * shell_disp[:,None,:]
            ecoul += 0.5 * tf.reduce_sum(Vij_qz2 * charges[:,None,None,None] * shell_d2[None,:,:,:])
            ecoul += 0.5 * tf.reduce_sum(Vij_zq2 * charges[None,:,None,None] * shell_d2[:,None,:,:])
            ecoul += tf.reduce_sum((E_d1  + 0.5 * E_qd * dq[:,None] + 0.5 *  E_d2 * shell_disp) * shell_disp)
            if self.apply_field:
                Piq_a, Pie_a = _ewald.atom_centered_polarization(shell_disp,
                                                             nuclei_charge,
                                                             charges,
                                                             self.central_atom_id,
                                                             atomic_number)
            else:
                Piq_a, Pie_a = tf.zeros(3), tf.zeros(3)
            Pi_a = Piq_a + Pie_a
            efield_energy = -tf.reduce_sum(Pi_a * self.efield)
            ecoul += efield_energy
            total_energy += ecoul
            Vj = _ewald.recip_space_term_with_shelld_linear_Vj(shell_disp,
                                               nuclei_charge,
                                               charges)

        #differentiating a scalar w.r.t tensors
        pad_rows = tf.zeros([nmax_diff], dtype=tf.float32)
        charges = tf.concat([charges, pad_rows], axis=0)
        #Pi_a = -tape0.gradient(total_energy, self.efield) #contains P0 - dE/d_eps
        #total_energy -= tf.reduce_sum(Pi_a * _efield)
        forces = tape0.gradient(total_energy, positions)
        #needs tape to be persistent
        dE_dh = tape0.gradient(total_energy, cell)
        V = tf.abs(tf.tensordot(cell[0], tf.linalg.cross(cell[1], cell[2]),axes=1))
        # Stress: stress_{ij} = (1/V) \sum_k dE/dh_{ik} * h_{jk}
        stress = tf.linalg.matmul(dE_dh, cell, transpose_b=True) / V
        pad_rows = tf.zeros([nmax_diff, tf.shape(forces)[1]], dtype=forces.dtype)
        forces = tf.concat([-forces, pad_rows], axis=0)
        shell_disp = tf.concat([shell_disp, pad_rows], axis=0)
        return [total_energy, forces, C6, charges, stress, shell_disp, Pi_a, E1, E2, E_d2,E_d1,E_qd, Vj]


    ###kept for hitorical reasons
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
    def tf_predict_energy_forces_original(self,x):
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
        #positions = tf.Variable(positions)
        #cell = tf.Variable(cell)
        if self.efield is not None:
            _efield = tf.cast(self.efield, tf.float32)

            apply_field = True
        else:
            _efield = tf.cast([0.0,0.0,0.0], tf.float32)
            apply_field = False
        #with tf.GradientTape(persistent=True) as tape1:
            #the computation of Zstar is a second derivative and require additional gradient tape recording when computing the forces
       #     tape1.watch(_efield)
        with tf.GradientTape(persistent=True) as tape0:
            #'''
            tape0.watch(positions)
            tape0.watch(cell)
            tape0.watch(_efield)

            #based on ase 
            #npairs x 3
            all_rij = tf.gather(positions,second_atom_idx) - \
                     tf.gather(positions,first_atom_idx) + \
                    tf.tensordot(shift_vector,cell,axes=1)

            all_rij_norm = tf.linalg.norm(all_rij, axis=-1) #npair
            '''
            #damping function for atomic chi and J
            fc_ij = help_fn.tf_fcut_rbf(all_rij_norm,rc=1.2)
            fc_ij = tf.scatter_nd(
                indices=tf.stack([first_atom_idx, second_atom_idx], axis=1),
                updates=fc_ij,
                shape=(nat, nat)
            )
            # the diagonal elements are currently zero since they are never present in the neighbor list
            #instead, they should be 1.0
            fc_ij += tf.eye(nat, dtype=tf.float32)
            #fc_ij = tf.eye(nat, dtype=tf.float32)
            '''
            reg = 1e-12
            #all_rij_norm = tf.sqrt(tf.reduce_sum(all_rij * all_rij , axis=-1) + reg) #npair
            species_encoder_i = tf.gather(species_encoder,first_atom_idx)
            species_encoder_j = tf.gather(species_encoder,second_atom_idx)
            if self.species_correlation=='tensor':
                species_encoder_extended = tf.expand_dims(species_encoder_i, -1) * tf.expand_dims(species_encoder_j, -2)

               # species_encoder_extended = tf.einsum('ik,il->ikl',
               #                                  tf.gather(species_encoder,first_atom_idx),
               #                                  tf.gather(species_encoder,second_atom_idx)
               #                                  )
            else:
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
            #radial_ij = tf.einsum('ijl,ik->ijkl',bf_radial, species_encoder_ij) # npairs x Nrad x nembeddingxzeta (l=zeta)
            #radial_ij = bf_radial[:,:,None,:] * species_encoder_ij[:,None,:,None]
            radial_ij = tf.expand_dims(bf_radial, 2) * tf.expand_dims(tf.expand_dims(species_encoder_ij, 1), -1)
            #radial_ij = tf.einsum('ijl,ik->ijkl',bf_radial, species_encoder_ij) # npairs x Nrad x nembeddingxzeta (l=zeta)
            radial_ij = tf.reshape(radial_ij, [num_pairs, self.Nrad*self.spec_size,self.number_radial_components])
            atomic_descriptors = tf.math.unsorted_segment_sum(data=radial_ij[:,:,0],
                                                              segment_ids=first_atom_idx, num_segments=nat) 

            #implement angular part: compute vect_rij dot vect_rik / rij / rik
            #rij_unit = tf.einsum('ij,i->ij',all_rij, 1.0 / (all_rij_norm+reg)) #npair,3
            rij_unit = all_rij / (tf.expand_dims(all_rij_norm + reg, -1))

            if self.body_order == 3:
                radial_ij_extended = tf.gather(radial_ij[:,:,:(1+self.zeta[0])], self.lxlylz_sum[0], axis=2)
                #Gi3 = self._to_three_body_terms(rij_unit, radial_ij_extended, first_atom_idx,nat)
                Gi3 = to_three_body_terms(rij_unit, radial_ij_extended, first_atom_idx,nat)
                #Gi = self._to_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat)
                #Gi3 = Gi[0]
                atomic_descriptors = tf.concat([atomic_descriptors, Gi3], axis=1)
            elif self.body_order == 4:
                #Gi3,Gi4 = self._to_four_body_terms(rij_unit, radial_ij, first_atom_idx, nat)
                Gi = to_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat, self.body_order)
                Gi3,Gi4 = Gi
                atomic_descriptors = tf.concat([atomic_descriptors, Gi3], axis=1)
                atomic_descriptors = tf.concat([atomic_descriptors, Gi4], axis=1)
            elif self.body_order == 5:
                #Gi3,Gi4,Gi5 = self._to_five_body_terms(rij_unit, radial_ij, first_atom_idx, nat)
                #Gi = self._to_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat)
                Gi = to_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat, self.body_order)
                Gi3,Gi4,Gi5 = Gi
                #self._to_five_body_terms(rij_unit, radial_ij, first_atom_idx, nat)
                atomic_descriptors = tf.concat([atomic_descriptors, Gi3], axis=1)
                atomic_descriptors = tf.concat([atomic_descriptors, Gi4], axis=1)
                atomic_descriptors = tf.concat([atomic_descriptors, Gi5], axis=1)
            
            #field components = \sum_j{fc(rij) * (E . rij_unit)}
            _efield_extended = tf.squeeze(tf.matmul(rij_unit, _efield[:, None])) * help_fn.tf_fcut_rbf(all_rij_norm, rc)
            _efield_extended = tf.math.unsorted_segment_sum(data=_efield_extended,
                                                              segment_ids=first_atom_idx, num_segments=nat)

            atomic_descriptors = tf.concat([atomic_descriptors, _efield_extended[:,None]], axis=1) # append electric field
            atomic_descriptors = tf.reshape(atomic_descriptors, [nat, self.feature_size])

            #predict energy and forces
            _atomic_energies = self.atomic_nets(atomic_descriptors) #nmax,ncomp aka head
            if self.coulumb or self.include_vdw:
                total_energy = tf.reduce_sum(_atomic_energies[:,0])
            else:
                total_energy = tf.reduce_sum(_atomic_energies)
            idx = 1
            if self.include_vdw:
                # C6 has a shape of nat
                C6 = tf.nn.relu(_atomic_energies[:,idx])
                C6_ij = tf.gather(C6,second_atom_idx) * tf.gather(C6,first_atom_idx) # npair,
                #na from C6_extended in 1xNeigh and C6 in nat
                C6_ij = tf.sqrt(C6_ij + 1e-16)
                evdw = help_fn.vdw_contribution((all_rij_norm, C6_ij,
                                                 self.rmin_u,
                                                 self.rmax_u,
                                                 self.rmin_d,
                                                 self.rmax_d))[0]
                total_energy += evdw

                C6 = tf.pad(C6,[[0,nmax_diff]])
                idx += 1
            else:
                C6 = tf.pad(C6,[[0,nmax_diff]])
            if self.coulumb:
                E1 = tf.nn.softplus(_atomic_energies[:,idx])
                #E1 = _atomic_energies[:,idx] 
                idx += 1
                E2 = tf.nn.softplus(_atomic_energies[:,idx])
                #E2 = _atomic_energies[:,idx] # can modulate the diagonal elements either ways
                if self.pqeq:
                    idx += 1
                    if self._anisotropy:
                        if self.linear_d_terms:
                            E_d1 = tf.reshape(tf.math.tanh(_atomic_energies[:,idx:idx+3]), [nat,3]) * 0.1 # eV/A
                            idx += 3
                            E_d2 = tf.reshape(tf.math.sigmoid(_atomic_energies[:,idx:idx+3]), [nat,3])  # eV/A^2
                            idx += 3
                            E_qd = tf.reshape(tf.math.tanh(_atomic_energies[:,idx:]), [nat,3]) * 0.1  # V/A
                        else:
                            E_d2 = tf.reshape(tf.nn.softplus(_atomic_energies[:,idx:]), [nat,3]) # eV/A^2
                            E_d1 = tf.zeros((nat,3))
                            E_qd = tf.zeros((nat,3))

                    else:
                        if self.linear_d_terms:
                            E_d1 = tf.tile(tf.math.tanh(_atomic_energies[:,idx])[:,None], [1,3]) * 0.1 # eV/A [-0.1,0.1]
                            idx += 1
                            E_d2 = 2.0 + tf.tile(tf.nn.softplus(_atomic_energies[:,idx])[:,None], [1,3]) # eV/A^2 [2,infty]
                            idx += 1
                            E_qd = tf.tile(tf.math.tanh(_atomic_energies[:,idx])[:,None], [1,3]) * 0.1 # eV/A [-0.1,0.1]
                        else:
                            E_d2 = 2.0 + tf.tile(tf.nn.softplus(_atomic_energies[:,idx])[:,None], [1,3]) # eV/A^2 [2,inf]
                            E_d1 = tf.zeros((nat,3))
                            E_qd = tf.zeros((nat,3))

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
                if not self.pqeq:
                    Vij = _ewald.recip_space_term() if self.pbc else _ewald.real_space_term()

                    if apply_field:
                        field_kernel, field_kernel_e = _ewald.potential_linearized_periodic_ref0(tf.zeros_like(nuclei_charge))
                        _b += field_kernel
                    charges = self.compute_charges(Vij, _b, E2, atomic_q0, total_charge)
                    ecoul = self.compute_coulumb_energy(charges, atomic_q0, E1, E2, Vij)
                else:
                    if apply_field:
                        if self._sawtooth_PE:
                            field_kernel_q, field_kernel_e, field_kernel_qe = _ewald.potential_linearized_sin(nuclei_charge)
                            #E_d2 += field_kernel_ed
                        elif self._P_in_cell:
                            if self._linearize_d == 0 or self._linearize_d == 1:
                                field_kernel_q, field_kernel_e = _ewald.potential_linearized_periodic_ref0(nuclei_charge)
                                field_kernel_qe = tf.zeros_like(field_kernel_e)
                            else:
                                field_kernel_q, field_kernel_e, field_kernel_qe = _ewald.potential_linearized_periodic_ref1(nuclei_charge)

                        else:
                            #field_kernel_q, field_kernel_e, field_kernel_qe = _ewald.atom_centered_dV_qs(nuclei_charge,
                            if self._linearize_d == 0 or self._linearize_d == 1:
                                field_kernel_q, field_kernel_e, field_kernel_qe = _ewald.atom_centered_dV(nuclei_charge,
                                                                                     self.central_atom_id, 
                                                                                     atomic_number)
                            else:
                                field_kernel_q, field_kernel_e, field_kernel_qe = _ewald.atom_centered_dV_qs(nuclei_charge,
                                                                                     self.central_atom_id, 
                                                                                     atomic_number)
                        _b += field_kernel_q # the term coming from qi-nuclei. The electronic contribution does not contribute to change in nuclei charges
                    else:
                        field_kernel_q,field_kernel_e, field_kernel_qe = tf.zeros(nat), tf.zeros((nat,3)), tf.zeros((nat,3))
                    
                    #E_d1 += field_kernel_e # the linear term in d
                    if self._linearize_d == 0:
                        _E_d1 = (E_d1 + field_kernel_e - 0.5 * E_qd * atomic_q0[:,None])
                        Vij, Vij_qz, Vij_zq, Vij_zz = _ewald.recip_space_term_with_shelld_quadratic_qd(nuclei_charge)
                        #E_d1 = field_kernel_e
                        charges, shell_disp = _compute_charges_disp(Vij, Vij_qz, Vij_zq, Vij_zz,
                             _b, E2, E_d2, _E_d1, E_qd + field_kernel_qe, # field_kernel_qe comes from placing Z at the nuclei position and q-Z at the shell
                             atomic_q0, total_charge)
                        ecoul = _compute_coulumb_energy_pqeq_qd(charges, atomic_q0,
                                    E1, E2, shell_disp, Vij, Vij_qz, Vij_zq, Vij_zz)
                        dq = charges - atomic_q0
                        ecoul += tf.reduce_sum((E_d1  + 0.5 * E_qd * dq[:,None] + 0.5 *  E_d2 * shell_disp) * shell_disp) 
                    elif self._linearize_d == 1:
                        #Vij = _ewald.recip_space_term() # in d=0 approximation determine charges
                        Vij, Vij_qz, Vij_zq, Vij_zz, Vij_qz2, Vij_zq2 = _ewald.recip_space_term_with_shelld_quadratic_qqdd_1(nuclei_charge)
                        # compute charges at d = 0
                        charges = _compute_charges(Vij, _b, E2, atomic_q0, total_charge)
                        # determine d
                        shell_disp = _compute_shell_disp_qqdd1(Vij_qz, Vij_zq, Vij_qz2, Vij_zq2, Vij_zz, E_d1,
                           E_d2 + field_kernel_qe, E_qd, atomic_q0,charges, field_kernel_e)

                        #update charges
                        shell_d2 = shell_disp[:,:,None] * shell_disp[:,None,:]
                        #_b += 0.5 * tf.reduce_sum((tf.transpose(Vij_zq,perm=(1,0,2)) + Vij_qz) * shell_disp[None,:,:], axis=(1,2)) #N
                        #_b += 0.5 * tf.reduce_sum((tf.transpose(Vij_zq2,perm=(1,0,2,3)) + Vij_qz2) * shell_d2[None,...], axis=(1,2,3)) #N
                        #_b += 0.5 * tf.reduce_sum(E_qd * shell_disp, axis=1) #sum over cartessian directions
                        #charges = _compute_charges(Vij, _b, E2, atomic_q0, total_charge)
                        
                        dq = charges - atomic_q0
                        
                        '''
                        #A_op = AOperator(A_iajb)
                        A_op = tf.linalg.LinearOperatorFullMatrix(
                        A_iajb,
                        is_self_adjoint=True,
                        is_positive_definite=True,  # Optional: set to True if you know it is
                        is_non_singular=True        # Optional: set to True if you know it is
                        )
                        # using conjugate gradient
                        # (uses atomic_q0_padded as initial guess x0)
                        cg_result = tf.linalg.experimental.conjugate_gradient(
                            operator=A_op,
                            rhs=-A_ia,
                            x=tf.zeros_like(A_ia),
                            tol=1e-6,
                            max_iter=100)
                        # Extract the solution vector (length N+1).
                        shell_disp = cg_result.x # make it a column vector
                        shell_disp = tf.reshape(shell_disp, [nat,3])
                        '''
                        ecoul = _compute_coulumb_energy_pqeq_qd(charges, atomic_q0,
                                    E1, E2, shell_disp, Vij, Vij_qz, Vij_zq, Vij_zz)
                        #add additional terms to energy
                        #shell_d2 = shell_disp[:,:,None] * shell_disp[:,None,:]
                        ecoul += 0.5 * tf.reduce_sum(Vij_qz2 * charges[:,None,None,None] * shell_d2[None,:,:,:])
                        ecoul += 0.5 * tf.reduce_sum(Vij_zq2 * charges[None,:,None,None] * shell_d2[:,None,:,:])
                        ecoul += tf.reduce_sum((E_d1  + 0.5 * E_qd * dq[:,None] + 0.5 *  E_d2 * shell_disp) * shell_disp) 
                    elif self._linearize_d == 2:
                        #too busy here, may worth  moving to a new function
                        V_mat = _ewald.recip_space_term_with_shelld_quadratic_qqdd_2(nuclei_charge)
                        Vij, Vij_qz, Vij_zq, Vij_zz, Vij_qz2, Vij_zq2, Vij_qq2, Vij_qq3 = V_mat
                        
                        charges = _compute_charges(Vij, _b, E2, atomic_q0, total_charge)
                        dq = charges - atomic_q0
                        shell_disp = _compute_shell_disp_qqdd2(Vij, Vij_qz, Vij_zq, Vij_zz,
                                 Vij_qz2, Vij_zq2, Vij_qq2, Vij_qq3, E_d1,
                           E_d2, E_qd, atomic_q0,charges, field_kernel_e,field_kernel_qe)
                        
                        # add other ijab terms to Vij_zz
                        charge_ij = charges[:,None] * charges[None,:]
                        Vij_zz += (-2. * Vij_qq3 * charge_ij[:,:,None,None] -
                                   2.0 * Vij_qz2 * charges[None,:,None,None] -
                                   2.0 * Vij_zq2 * charges[:,None,None,None])
                        ecoul = _compute_coulumb_energy_pqeq_qd(charges, atomic_q0,
                                    E1, E2, shell_disp, Vij, Vij_qz, Vij_zq, Vij_zz)
                        #add additional terms
                        shell_d2 = shell_disp[:,:,None] * shell_disp[:,None,:]
                        ecoul += 0.5 * tf.reduce_sum(Vij_qz2 * charges[:,None,None,None] * shell_d2[None,:,:,:])
                        ecoul += 0.5 * tf.reduce_sum(Vij_zq2 * charges[None,:,None,None] * shell_d2[:,None,:,:])
                        
                        dij = shell_disp[None,:,:] - shell_disp[:,None,:]
                        ecoul += 0.5 * tf.reduce_sum(Vij_qq2 * charge_ij[:,:,None] * dij)
                        sum_shell_d2 = shell_d2[:,None,:,:] + shell_d2[None,:,:,:]
                        ecoul += 0.5 * tf.reduce_sum(Vij_qq3 * charge_ij[:,:,None,None] * sum_shell_d2)
                        ecoul += tf.reduce_sum((E_d1  + 0.5 * E_qd * dq[:,None] + 0.5 *  E_d2 * shell_disp) * shell_disp) 
                    else:
                        shell_disp = tf.ones((nat,3)) * 1e-3
                        Vij, Vij_qz, Vij_zq, Vij_zz = _ewald.recip_space_term_with_shelld(shell_disp)
                        #Vij = _ewald.recip_space_term() # in d=0 approximation
                        #determine charges
                        _b += 0.5 * tf.reduce_sum((Vij_qz + tf.transpose(Vij_zq)) * nuclei_charge[None,:], axis=1)
                        charges = _compute_charges(Vij, _b, E2, atomic_q0, total_charge)

                        shell_disp = _ewald.shell_optimization_newton(shell_disp, nuclei_charge, 
                                                                      charges, E_d2, field_kernel_e,
                                                                      max_iter=1, tol=1e-3)
                        Vij, Vij_qz, Vij_zq, Vij_zz = _ewald.recip_space_term_with_shelld(shell_disp)
                        ecoul = _compute_coulumb_energy_pqeq(charges, atomic_q0, nuclei_charge,
                                    E1, E2, Vij, Vij_qz, Vij_zq, Vij_zz)
                        ecoul += tf.reduce_sum(0.5 *  E_d2 * shell_disp * shell_disp) 
                if self._P_in_cell:
                     Piq_a = tf.reduce_sum((charges + nuclei_charge)[:,None]  * positions, axis=0)
                     Pie_a = -tf.reduce_sum(nuclei_charge[:,None] * (positions + shell_disp), axis=0)
                 #    Pi_a = Piq_a + Pie_a
                elif self._sawtooth_PE:
                     Piq_a, Pie_a = _ewald.polarization_linearized_sin(nuclei_charge, charges, shell_disp)
                #     Pi_a = Piq_a + Pie_a
                else:
                    if self._linearize_d == 0 or self._linearize_d == 1:
                        Piq_a, Pie_a = _ewald.atom_centered_polarization(shell_disp,
                                                                     nuclei_charge,
                                                                     charges,
                                                                     self.central_atom_id,
                                                                     atomic_number
                                                                     )
                    else:
                        Piq_a, Pie_a = _ewald.atom_centered_polarization_qs(shell_disp,
                                                                     nuclei_charge,
                                                                     charges,
                                                                     self.central_atom_id,
                                                                     atomic_number
                                                                     )

                #Pi_a = tf.stack([Piq_a + Pie_a, Piq_a, Pie_a])
                Pi_a = Piq_a + Pie_a
                if apply_field:
                    if not self.pqeq:
                        efield_energy = tf.reduce_sum(charges * field_kernel)
                        ecoul += efield_energy
                    else:
                        #both nuclei and electron
                        efield_energy = -tf.reduce_sum((Piq_a + Pie_a) * _efield)
                        ecoul += efield_energy 

                Vj = _ewald.recip_space_term_with_shelld_linear_Vj(shell_disp,
                                               nuclei_charge,
                                               charges)
                total_energy += ecoul
            else:
                charges = tf.zeros([nmax_diff+nat], dtype=tf.float32)
                Vj = tf.zeros(nat, dtype=tf.float32)

        #differentiating a scalar w.r.t tensors
        pad_rows = tf.zeros([nmax_diff], dtype=tf.float32)
        charges = tf.concat([charges, pad_rows], axis=0)
        Pi_a = -tape0.gradient(total_energy, _efield) #contains P0 - dE/d_eps
        #total_energy -= tf.reduce_sum(Pi_a * _efield)
        forces = tape0.gradient(total_energy, positions)
        '''
        #compute zstar as dP_a/dRi_b

        if self.coulumb and self.efield is not None:
            #zstar = tf.transpose(tape0.jacobian(P_total, positions), [1,0,2]) # [3,Nat,3]
            #zstar = tape0.jacobian(forces, _efield) # [Nat,3,3]
            zstar = tf.zeros((nat,3,3), dtype=tf.float32)
        else:
            zstar = tf.zeros((nat,3,3), dtype=tf.float32)
        '''
        #needs tape to be persistent
        if self.pbc:
            dE_dh = tape0.gradient(total_energy, cell)
            #V = tf.abs(tf.linalg.det(cell))
            V = tf.abs(tf.tensordot(cell[0], tf.linalg.cross(cell[1], cell[2]),axes=1))
            # Stress: stress_{ij} = (1/V) \sum_k dE/dh_{ik} * h_{jk}
            stress = tf.linalg.matmul(dE_dh, cell, transpose_b=True) / V
        else:
            stress = tf.zeros((3,3), dtype=tf.float32)
        #born effective charges for direct differentiation
        #if self.efield is not None and not self.is_training:
        #    Zstar = tf.squeeze(tape1.jacobian(forces,_efield, experimental_use_pfor=True)) #This is in unit of electron charges
        #    Zstar = tf.reshape(Zstar, [nat,9])
        #    Zstar = tf.pad(-Zstar, paddings=[[0,nmax_diff],[0,0]])
            #Zstar = tf.reshape(Zstar, [-1])
        #else:
        #    Zstar = tf.zeros((nat+nmax_diff,9))
        pad_rows = tf.zeros([nmax_diff, tf.shape(forces)[1]], dtype=forces.dtype)
        forces = tf.concat([-forces, pad_rows], axis=0)
        shell_disp = tf.concat([shell_disp, pad_rows], axis=0)

        #forces = tf.pad(-forces, paddings=[[0,nmax_diff],[0,0]], constant_values=0.0)
        return [total_energy, forces, C6, charges, stress, shell_disp, Pi_a, E1, E2, E_d2,E_d1,E_qd, Vj]
