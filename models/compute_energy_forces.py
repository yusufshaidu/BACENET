import jax
import jax.numpy as jnp
from jax import lax

def jax_predict_energy_forces(self, rc, Nrad, x):

    nat = x[3]
    nmax_diff = x[2]

    species_encoder = jnp.reshape(x[0][:nat * self.nspec_embedding],
                                  (nat, self.nspec_embedding))
    positions = jnp.reshape(x[1][:nat*3], (nat, 3))
    cell = jnp.reshape(x[4], (3,3))
    C6 = x[5][:nat]
    num_pairs = x[9]
    first_atom_idx = x[6][:num_pairs].astype(jnp.int32)
    second_atom_idx = x[7][:num_pairs].astype(jnp.int32)
    shift_vector = jnp.reshape(x[8][:num_pairs*3], (num_pairs, 3)).astype(jnp.float32)

    gaussian_width = jnp.reshape(x[10][:nat*2], (nat, 2))
    chi0 = x[11][:nat]
    J0 = x[12][:nat]
    atomic_q0 = x[13][:nat]
    total_charge = x[14]
    nuclei_charge = x[15][:nat]
    atomic_number = x[16][:nat]

    _efield = jnp.array(self.efield if self.efield is not None else [0.,0.,0.], dtype=jnp.float32)
    apply_field = self.efield is not None

    reg = 1e-12

    def energy_fn(positions, cell):
        # -----------------------------
        # Interatomic distances & radial functions
        # -----------------------------
        all_rij = jnp.take(positions, second_atom_idx, axis=0) - \
                  jnp.take(positions, first_atom_idx, axis=0) + \
                  jnp.dot(shift_vector, cell)
        all_rij_norm = jnp.linalg.norm(all_rij, axis=-1)

        species_encoder_i = jnp.take(species_encoder, first_atom_idx, axis=0)
        species_encoder_j = jnp.take(species_encoder, second_atom_idx, axis=0)
        if self.species_correlation == 'tensor':
            species_encoder_extended = species_encoder_i[..., None] * species_encoder_j[:, None, :]
        else:
            species_encoder_extended = species_encoder_i * species_encoder_j
        species_encoder_ij = jnp.reshape(species_encoder_extended, (-1, self.spec_size))

        kn_rad = jnp.ones(self.n_bessels, dtype=jnp.float32)
        bf_radial0 = help_fn.bessel_function(all_rij_norm, rc, kn_rad, self.n_bessels)
        bf_radial1 = jnp.reshape(bf_radial0, (-1, self.n_bessels))
        bf_radial2 = self.radial_funct_net(bf_radial1)
        bf_radial = jnp.reshape(bf_radial2, (num_pairs, self.Nrad, self.number_radial_components))
        radial_ij = bf_radial[:, :, None, :] * species_encoder_ij[:, None, None, :]
        radial_ij = jnp.reshape(radial_ij, (num_pairs, self.Nrad * self.spec_size, self.number_radial_components))
        atomic_descriptors = jax.ops.segment_sum(radial_ij[:,:,0], first_atom_idx, nat)

        rij_unit = all_rij / (all_rij_norm[:, None] + reg)

        # -----------------------------
        # Body order terms
        # -----------------------------
        if self.body_order == 3:
            radial_ij_extended = jnp.take(radial_ij[:,:,:(1+self.zeta[0])], self.lxlylz_sum[0], axis=2)
            Gi3 = self._to_three_body_terms(rij_unit, radial_ij_extended, first_atom_idx, nat)
            atomic_descriptors = jnp.concatenate([atomic_descriptors, Gi3], axis=1)
        else:
            Gi = self._to_body_order_terms(rij_unit, radial_ij, first_atom_idx, nat)
            for Gi_part in Gi:
                atomic_descriptors = jnp.concatenate([atomic_descriptors, Gi_part], axis=1)

        atomic_descriptors = jnp.reshape(atomic_descriptors, (nat, self.feature_size))

        # -----------------------------
        # Atomic energy
        # -----------------------------
        _atomic_energies = self.atomic_nets(atomic_descriptors)
        total_energy = jnp.sum(_atomic_energies[:,0]) if (self.coulumb or self.include_vdw) else jnp.sum(_atomic_energies)

        idx = 1
        evdw = 0.0
        if self.include_vdw:
            C6 = jax.nn.relu(_atomic_energies[:, idx])
            C6_ij = jnp.sqrt(jnp.take(C6, second_atom_idx) * jnp.take(C6, first_atom_idx) + 1e-16)
            evdw = help_fn.vdw_contribution((all_rij_norm, C6_ij,
                                             self.rmin_u, self.rmax_u, self.rmin_d, self.rmax_d))[0]
            total_energy += evdw
            idx += 1

        # -----------------------------
        # Coulomb + PQEq + Shell Displacement
        # -----------------------------
        if self.coulumb:
            # Softplus energies for charges
            E1 = jax.nn.softplus(_atomic_energies[:, idx])
            idx += 1
            E2 = jax.nn.softplus(_atomic_energies[:, idx])
            idx += 1
            if self.pqeq:
                # Anisotropic shell
                if self._anisotropy:
                    if self.linear_d_terms:
                        E_d1 = jnp.reshape(jnp.tanh(_atomic_energies[:, idx:idx+3]), (nat,3)) * 0.1
                        idx += 3
                        E_d2 = jnp.reshape(jax.nn.sigmoid(_atomic_energies[:, idx:idx+3]), (nat,3)) * 5
                        idx += 3
                        E_qd = jnp.reshape(jnp.tanh(_atomic_energies[:, idx:idx+3]), (nat,3)) * 0.1
                        idx += 3
                    else:
                        E_d2 = jnp.reshape(jax.nn.sigmoid(_atomic_energies[:, idx:]), (nat,3))
                        E_d1 = jnp.zeros((nat,3))
                        E_qd = jnp.zeros((nat,3))
                else:
                    if self.linear_d_terms:
                        E_d1 = jnp.tile(jnp.tanh(_atomic_energies[:,idx])[:,None], (1,3)) * 0.1
                        idx += 1
                        E_d2 = jnp.tile(jax.nn.sigmoid(_atomic_energies[:,idx])[:,None], (1,3)) * 5
                        idx += 1
                        E_qd = jnp.tile(jnp.tanh(_atomic_energies[:,idx])[:,None], (1,3)) * 0.1
                        idx += 1
                    else:
                        E_d2 = jnp.tile(jax.nn.sigmoid(_atomic_energies[:,idx])[:,None], (1,3)) * 5
                        E_d1 = jnp.zeros((nat,3))
                        E_qd = jnp.zeros((nat,3))

            # Include electronegativity and hardness
            E1 += chi0
            E2 += J0
            _b = E1 - E2 * atomic_q0

            # -----------------------------
            # Ewald computation
            # -----------------------------
            _ewald = ewald(positions, cell, nat, gaussian_width,
                           self.accuracy, None, self.pbc, _efield,
                           self.gaussian_width_scale)

            if not self.pqeq:
                Vij = _ewald.recip_space_term() if self.pbc else _ewald.real_space_term()
                if apply_field:
                    field_kernel, field_kernel_e = _ewald.potential_linearized_periodic_ref0(jnp.zeros_like(nuclei_charge))
                    _b += field_kernel
                charges = self.compute_charges(Vij, _b, E2, atomic_q0, total_charge)
                ecoul = self.compute_coulumb_energy(charges, atomic_q0, E1, E2, Vij)
            else:
                # PQEq, shell displacement
                # Choose linearize_d method 0/1/2
                charges, shell_disp, ecoul = self._compute_pqeq_energy(_ewald, _b, E1, E2,
                                                                       E_d1, E_d2, E_qd,
                                                                       atomic_q0, total_charge,
                                                                       nuclei_charge, apply_field)

            total_energy += ecoul

        return total_energy

    # -----------------------------
    # Compute energy + forces
    # -----------------------------
    total_energy, forces_fn = jax.value_and_grad(energy_fn, argnums=0)(positions, cell)
    forces = -forces_fn

    if self.pbc:
        dE_dh = jax.grad(energy_fn, argnums=1)(positions, cell)
        V = jnp.abs(jnp.dot(cell[0], jnp.cross(cell[1], cell[2])))
        stress = jnp.dot(dE_dh, cell.T) / V
    else:
        stress = jnp.zeros((3,3), dtype=jnp.float32)

    pad_rows = jnp.zeros((nmax_diff, 3), dtype=jnp.float32)
    forces = jnp.concatenate([forces, pad_rows], axis=0)

    # Return structure same as TF version
    return [total_energy, forces, C6, jnp.zeros_like(C6), stress,
            jnp.zeros_like(forces), jnp.zeros((3,3)), jnp.zeros(nat), jnp.zeros(nat),
            jnp.zeros((nat,3)), jnp.zeros((nat,3)), jnp.zeros((nat,3)), jnp.zeros(nat)]
