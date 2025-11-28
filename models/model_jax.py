import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import logging


class BACENET(nn.Module):
    configs: dict

    # -------------------------------------------------------
    # Flax requires parameter creation in setup(), not __init__
    # -------------------------------------------------------
    def setup(self):
        configs = self.configs

        # -----------------------
        # Basic training settings
        # -----------------------
        self.is_training   = configs['is_training']
        self.starting_step = configs['initial_global_step']

        self.species_layer_sizes = configs['species_layer_sizes']
        layer_sizes = configs['layer_sizes']
        _activations = configs['activations']
        _radial_layer_sizes = list(configs['radial_layer_sizes'])

        # -----------------------
        # Feature parameters
        # -----------------------
        self.rcut   = configs['rc_rad']
        self.Nrad   = int(configs['Nrad'])
        self.n_bessels = configs['n_bessels'] if configs['n_bessels'] is not None else self.Nrad
        self.n_bessels = int(self.n_bessels)

        self.zeta = configs['zeta']  # list of max l per body order
        self.nzeta = sum(self.zeta)

        self.body_order = configs['body_order']

        base_size = self.Nrad * (self.zeta[0] + 1)
        self.feature_size = self.Nrad + base_size

        if 4 <= self.body_order < 8:
            i = 1
            for b in range(4, self.body_order + 1):
                self.feature_size += self.Nrad * (1 + self.zeta[i]) ** (b - 2)
                i += 1

        if self.body_order == 40:
            self.feature_size += self.Nrad * (self.zeta[1] + 1) ** 3

        # ------------------------------------
        # Species/embedding parameters
        # ------------------------------------
        self.species_correlation = configs['species_correlation']
        self.species_identity = configs['species_identity']
        self.nspecies = len(self.species_identity)
        self.species = configs['species']

        self.nspec_embedding = self.species_layer_sizes[-1]
        if self.species_correlation == 'tensor':
            self.spec_size = self.nspec_embedding * self.nspec_embedding
        else:
            self.spec_size = self.nspec_embedding

        self.feature_size *= self.spec_size

        logging.info(f"input dimension for the network features is {self.feature_size}")

        # --------------
        # Costs & losses
        # --------------
        self.batch_size = configs['batch_size']
        self.fcost      = float(configs['fcost'])
        self.ecost      = float(configs['ecost'])

        self.fcost_swa = float(configs['fcost_swa'])
        self.ecost_swa = float(configs['ecost_swa'])
        self._start_swa = configs['start_swa_global_step']

        # Norms
        self.l1 = float(configs['l1_norm'])
        self.l2 = float(configs['l2_norm'])

        self.learn_radial = configs['learn_radial']

        # ---------------------------
        # Dispersion, electrostatics
        # ---------------------------
        self.include_vdw = configs['include_vdw']
        self.rmin_u = configs['rmin_u']
        self.rmax_u = configs['rmax_u']
        self.rmin_d = configs['rmin_d']
        self.rmax_d = configs['rmax_d']

        self.nelement = configs['nelement']
        self.coulumb  = configs['coulumb']
        self.efield   = configs['efield']
        self._sawtooth_PE = configs['sawtooth_PE']
        self._P_in_cell   = configs['P_in_cell']

        print("This is P in cell status", self._P_in_cell)

        self.accuracy = configs['accuracy']
        self.pbc = configs['pbc']
        self.central_atom_id = configs['central_atom_id']
        self.species_nelectrons = configs['species_nelectrons']

        # ----------------------------------
        # Species electrons (JAX replacement)
        -----------------------------------
        if self.species_nelectrons is None:
            if configs['nshells'] == 0:
                self.species_nelectrons = np.array([2.0] * len(self.species))
            elif configs['nshells'] == -1:
                self.species_nelectrons = np.array(
                    [help_fn.unfilled_orbitals(symb) for symb in self.species]
                )
            else:
                self.species_nelectrons = np.array(
                    [help_fn.valence_with_two_shells(symb, configs['nshells'])
                     for symb in self.species]
                )

        self.species_nelectrons = jnp.array(self.species_nelectrons, dtype=jnp.float32)

        # ----------------------------
        # Oxidation states
        # ----------------------------
        self.oxidation_states = configs['oxidation_states']
        if self.oxidation_states is None:
            self.oxidation_states = jnp.zeros(len(self.species), dtype=jnp.float32)
        else:
            self.oxidation_states = jnp.array(self.oxidation_states, dtype=jnp.float32)

        mse_os = jnp.sum(self.oxidation_states ** 2)
        self.learn_oxidation_states = not (mse_os < 1e-3 or configs['qcost'] == 0.0)

        self.species_chi0 = configs['species_chi0'] * configs['scale_chi0']
        self.species_J0 = configs['species_J0'] * configs['scale_J0']

        self._max_width = configs['max_width']
        self._linearize_d = configs['linearize_d']
        self._anisotropy = configs['anisotropy']
        self.linear_d_terms = configs['linear_d_terms']
        self._d0 = configs['d0']

        if self.linear_d_terms and self._anisotropy:
            if layer_sizes[-1] != 12:
                raise ValueError("The last layer must be 12")
        elif self.linear_d_terms and not self._anisotropy:
            if layer_sizes[-1] != 6:
                raise ValueError("The last layer must be 6")

        if self._anisotropy and layer_sizes[-1] != 12:
            raise ValueError("The last layer must be 12")

        # --------------------
        # Feature selection
        # --------------------
        self.features = configs['features']
        self.coulumb_flag = configs['coulumb']

        self.pqeq = configs['pqeq']

        # --------------------
        # Atomic/bead networks
        # --------------------
        self.atomic_nets = Networks(
            self.feature_size,
            layer_sizes,
            _activations,
            l1=self.l1,
            l2=self.l2,
            normalize=configs['normalize'],
        )

        self._normalize = configs['normalize']

        # --------------------------
        # Species embedding network
        --------------------------
        species_activations = ['tanh'] * (len(self.species_layer_sizes) - 1)
        species_activations.append('linear')

        nelements = self.nelement + (1 if self.coulumb else 0)

        self.species_nets = Networks(
            nelements,
            self.species_layer_sizes,
            species_activations,
            prefix='species_encoder',
        )

        # ----------------------
        # Radial network
        # ----------------------
        b_order = self.body_order if self.body_order != 40 else 4

        self.number_radial_components = 1
        for i, z in enumerate(self.zeta):
            self.number_radial_components += (i + 1) * z + 1

        _radial_layer_sizes.append(self.number_radial_components * self.Nrad)

        radial_activations = ['silu'] * (len(_radial_layer_sizes) - 1)
        radial_activations.append('silu')

        self.radial_funct_net = Networks(
            self.n_bessels,
            _radial_layer_sizes,
            radial_activations,
            prefix='radial-functions',
        )

        # ---------------------------------------------------------------------
        # OPTIONAL: Gaussian width network / learnable electron count per species
        # ---------------------------------------------------------------------
        self.learnable_gaussian_width = configs['learnable_gaussian_width']
        if self.learnable_gaussian_width:
            self.gaussian_width_net = Networks(
                self.nelement,
                [64, 2],
                ['sigmoid', 'sigmoid'],
                prefix='species_gaussian_width',
            )

        self._learn_species_nelectrons = configs['learn_species_nelectrons']
        if self._learn_species_nelectrons:
            self.species_nelectrons_net = Networks(
                self.nelement,
                [64, 1],
                ['softplus', 'softplus'],
                prefix='species_nelectrons',
            )

        # ------------------------------
        # Precompute cosine terms (JAX)
        # ------------------------------
        self.lxlylz = []
        self.lxlylz_sum = []
        self.fact_norm = []

        for z in self.zeta:
            lxlylz, lxlylz_sum, fact_norm = help_fn._compute_cosine_terms(z)

            self.lxlylz.append(jnp.array(lxlylz, dtype=jnp.float32))
            self.lxlylz_sum.append(jnp.array(lxlylz_sum, dtype=jnp.int32))
            self.fact_norm.append(jnp.array(fact_norm, dtype=jnp.float32))
    
    def segment_sum(data, segment_ids, num_segments):
        return jax.ops.segment_sum(data, segment_ids, num_segments)
    
    def _angular_terms(self, rij_unit, lxlylz):
        """
        rij_unit: [npairs, 3]
        lxlylz:  [n_lxlylz, 3]
        Returns: [npairs, n_lxlylz]
        """
        # (npairs,1,3) ** (1,n_lxlylz,3)
        rij_lxlylz = (rij_unit[:, None, :] + 1e-12) ** (lxlylz[None, :, :])

        # Multiply x^lx * y^ly * z^lz
        g_ij_lxlylz = rij_lxlylz[:, :, 0] * rij_lxlylz[:, :, 1] * rij_lxlylz[:, :, 2]

        return g_ij_lxlylz
    def _to_three_body_terms(self, rij_unit, radial_ij, first_atom_idx, nat):
        """
        3-body symmetry functions.
        """

        # -----------------------
        # Angular terms
        # -----------------------
        g_ij_lxlylz = self._angular_terms(rij_unit, self.lxlylz[0]) 
        # shape = [npair, n_lxlylz]

        # -----------------------
        # Multiply radial * angular
        # -----------------------
        # radial_ij: [npair, nspec*nrad, nzeta]
        g_ilxlylz = radial_ij * g_ij_lxlylz[:, None, :]

        # Sum over neighbors j → central atom i
        g_ilxlylz = segment_sum(g_ilxlylz, first_atom_idx, nat)
        # shape = [nat, nspec*nrad, n_lxlylz]

        # Multiply and normalize
        _gi3 = (g_ilxlylz * g_ilxlylz).transpose(2, 0, 1) * self.fact_norm[0][:, None, None]
        # shape = [n_lxlylz, nat, nspec*nrad]

        gi3 = segment_sum(_gi3, self.lxlylz_sum[0], self.zeta[0] + 1)
        # shape = [nzeta, nat, nspec*nrad]

        gi3 = gi3.transpose(1, 0, 2)   # [nat, nzeta, nspec*nrad]

        return gi3.reshape(nat, -1)

    def _to_body_order_terms(self, rij_unit, radial_ij, first_atom_idx, nat):
        """
        up to 5-body symmetry functions.
        """

        # =========================
        # ----- 3-BODY TERMS ------
        # =========================
        g_ij_lxlylz = self._angular_terms(rij_unit, self.lxlylz[0])
        npairs, n_lxlylz = g_ij_lxlylz.shape

        r_start = 1
        r_end   = r_start + 1 + self.zeta[0]

        radial_ij_expanded = jnp.take(radial_ij[:, :, r_start:r_end],
                                      self.lxlylz_sum[0],
                                      axis=2)

        g_ilxlylz = radial_ij_expanded * g_ij_lxlylz[:, None, :]
        g_ilxlylz = segment_sum(g_ilxlylz, first_atom_idx, nat)

        _gi3 = (g_ilxlylz * g_ilxlylz).transpose(2, 0, 1) * self.fact_norm[0][:, None, None]

        gi3 = segment_sum(_gi3, self.lxlylz_sum[0], self.zeta[0] + 1)
        gi3 = gi3.transpose(1, 0, 2).reshape(nat, -1)

        if self.body_order == 3:
            return [gi3]

        # =========================
        # ----- 4-BODY TERMS ------
        # =========================
        g_ij_lxlylz = self._angular_terms(rij_unit, self.lxlylz[1])
        npairs, n_lxlylz = g_ij_lxlylz.shape

        r_start = r_end
        r_end = r_start + 1 + self.zeta[1]

        radial_ij_expanded = jnp.take(radial_ij[:, :, r_start:r_end],
                                      self.lxlylz_sum[1],
                                      axis=2)

        g_ilxlylz = radial_ij_expanded * g_ij_lxlylz[:, None, :]
        g_ilxlylz = segment_sum(g_ilxlylz, first_atom_idx, nat)

        g_i_l1l2 = g_ilxlylz[:, :, :, None] * g_ilxlylz[:, :, None, :]
        g_i_l1l2 = g_i_l1l2.reshape(nat, -1, n_lxlylz * n_lxlylz)

        r_end = r_start + 2 * self.zeta[1] + 1

        lxlylz_sum2 = (self.lxlylz_sum[1][:, None] + self.lxlylz_sum[1][None, :]).reshape(-1)
        fact_norm2  = (self.fact_norm[1][:, None] * self.fact_norm[1][None, :]).reshape(-1)

        radial_ij_expanded = jnp.take(radial_ij[:, :, r_start:r_end],
                                      lxlylz_sum2,
                                      axis=2)

        g_ij_l1_plus_l2 = (g_ij_lxlylz[:, :, None] *
                           g_ij_lxlylz[:, None, :]).reshape(-1, n_lxlylz * n_lxlylz)

        g_ij_ll = radial_ij_expanded * g_ij_l1_plus_l2[:, None, :]

        g_i_l1_plus_l2 = segment_sum(g_ij_ll, first_atom_idx, nat)

        g_i_l1l2_ijk = (g_i_l1l2 * g_i_l1_plus_l2).transpose(2, 0, 1) * fact_norm2[:, None, None]

        nzeta2 = (1 + self.zeta[1]) * (1 + self.zeta[1])

        g_i_l1l2 = segment_sum(g_i_l1l2_ijk, lxlylz_sum2, nzeta2)
        gi4 = g_i_l1l2.transpose(1, 0, 2).reshape(nat, -1)

        if self.body_order == 4:
            return [gi3, gi4]

        # =========================
        # ----- 5-BODY TERMS ------
        # =========================
        g_ij_lxlylz = self._angular_terms(rij_unit, self.lxlylz[2])
        npairs, n_lxlylz = g_ij_lxlylz.shape

        r_start = r_end
        r_end   = r_start + self.zeta[2] + 1

        radial_ij_expanded = jnp.take(radial_ij[:, :, r_start:r_end],
                                      self.lxlylz_sum[2],
                                      axis=2)

        g_ilxlylz = radial_ij_expanded * g_ij_lxlylz[:, None, :]
        g_ilxlylz = segment_sum(g_ilxlylz, first_atom_idx, nat)

        g_i_l1l2l3 = (
            g_ilxlylz[:, :, :, None, None] *
            g_ilxlylz[:, :, None, :, None] *
            g_ilxlylz[:, :, None, None, :]
        ).reshape(nat, -1, n_lxlylz**3)

        r_end = r_start + 3 * self.zeta[2] + 1

        lxlylz_sum3 = (
            self.lxlylz_sum[2][:, None, None] +
            self.lxlylz_sum[2][None, :, None] +
            self.lxlylz_sum[2][None, None, :]
        ).reshape(-1)

        fact_norm3 = (
            self.fact_norm[2][:, None, None] *
            self.fact_norm[2][None, :, None] *
            self.fact_norm[2][None, None, :]
        ).reshape(-1)

        radial_ij_expanded = jnp.take(radial_ij[:, :, r_start:r_end],
                                      lxlylz_sum3,
                                      axis=2)

        g_ij_l123 = (
            g_ij_lxlylz[:, :, None, None] *
            g_ij_lxlylz[:, None, :, None] *
            g_ij_lxlylz[:, None, None, :]
        ).reshape(-1, n_lxlylz**3)

        g_ij_lll = radial_ij_expanded * g_ij_l123[:, None, :]

        g_i_l123 = segment_sum(g_ij_lll, first_atom_idx, nat)

        g_i_l1l2l3_ijk = (g_i_l1l2l3 * g_i_l123).transpose(2, 0, 1) * fact_norm3[:, None, None]

        nzeta3 = (self.zeta[2] + 1)**3

        g_i_l1l2l3 = segment_sum(g_i_l1l2l3_ijk, lxlylz_sum3, nzeta3)
        gi5 = g_i_l1l2l3.transpose(1, 0, 2).reshape(nat, -1)

        return [gi3, gi4, gi5]

    def compute_Aij(self, Vij: jnp.ndarray, E2: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            Vij: 2D array of shape (N, N)
            E2: 1D array of shape (N,)

        Returns:
            Aij: (N+1, N+1) matrix
        """
        N = E2.shape[0]

        # Pad E2 with -1.0
        E2_padded = jnp.concatenate([E2, jnp.array([-1.0], dtype=E2.dtype)], axis=0)

        # Append a column of ones to Vij → shape (N, N+1)
        Aij_top = jnp.concatenate([Vij, jnp.ones((N, 1), dtype=Vij.dtype)], axis=1)

        # Create final row of ones → shape (1, N+1)
        bottom_row = jnp.ones((1, N + 1), dtype=Vij.dtype)

        # Stack to form (N+1, N+1)
        Aij = jnp.concatenate([Aij_top, bottom_row], axis=0)

        # Add diagonal(E2 padded)
        Aij = Aij + jnp.diag(E2_padded)

        return Aij

    def compute_Fia(self, Vij_qz: jnp.ndarray, E_qd: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            Vij_qz: shape (N, N, 3)
            E_qd: shape (N, 3)
        
        Returns:
            Fija: shape (N+1, 3*N)
        """
        N = Vij_qz.shape[0]

        # Identity delta_ij: shape (N, N)
        delta_ij = jnp.eye(N, dtype=Vij_qz.dtype)

        # Start with half Vij_qz
        Fija = 0.5 * Vij_qz  # shape (N, N, 3)

        # Expand E_qd along N-axis for elementwise multiplication
        E_expanded = E_qd[:, None, :] * delta_ij[:, :, None]  # shape (N, N, 3)

        Fija = Fija + 0.5 * E_expanded

        # Flatten last dimension into N*3 and pad one extra row at bottom
        Fija_flat = jnp.reshape(Fija, (N, N*3))
        Fija_padded = jnp.pad(Fija_flat, ((0, 1), (0, 0)))  # pad bottom row

        return Fija_padded  # shape (N+1, 3*N)

    def compute_Fiajb(self, Vij_zz: jnp.ndarray, E_di: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            Vij_zz: shape (N, N, 3, 3)
            E_di: shape (N, 3)

        Returns:
            Fiajb: shape (3*N, 3*N)
        """
        N = E_di.shape[0]

        # Identity matrices for atoms and Cartesian components
        delta_ij = jnp.eye(N, dtype=Vij_zz.dtype)           # shape (N, N)
        delta_ab = jnp.eye(3, dtype=Vij_zz.dtype)          # shape (3, 3)

        # Broadcast E_di along ijab axes
        Eijab = E_di[:, None, :, None] * delta_ij[:, :, None, None] * delta_ab[None, None, :, :]

        # Add Vij_zz and Eijab, then transpose axes to (i,a,j,b)
        Fiajb = jnp.transpose(Vij_zz + Eijab, (0, 2, 1, 3))  # shape (N, 3, N, 3)

        # Reshape into 2D matrix
        return jnp.reshape(Fiajb, (3*N, 3*N))

    def compute_charges_disp(self, Vij, Vij_qz, Vij_zq, Vij_zz,
                             E1, E2, E_d2, E_d1, E_qd,
                             atomic_q0, total_charge):
        """
        Compute charges and shell displacements via solution of linear system.
        """
        Vij_qz = Vij_qz + jnp.transpose(Vij_zq, (1, 0, 2))
        N = atomic_q0.shape[0]

        # Build block matrices
        Aij = jax.jit(compute_Aij)(Vij, E2)
        Fija = jax.jit(compute_Fia)(Vij_qz, E_qd)
        Fiajb = jax.jit(compute_Fiajb)(Vij_zz, E_d2)

        # Construct full system
        upper_layer = jnp.concatenate([Aij, Fija], axis=1)
        lower_layer = jnp.concatenate([Fija.T, Fiajb], axis=1)
        Mat = jnp.concatenate([upper_layer, lower_layer], axis=0)

        # Build RHS
        E1_padded = jnp.concatenate([-E1, jnp.array([total_charge], dtype=E1.dtype)], axis=0)
        b = jnp.concatenate([E1_padded, -E_d1.reshape(-1)], axis=0)

        # Solve linear system
        charges_disp = jnp.linalg.solve(Mat, b[:, None]).squeeze()

        # Split outputs
        charges = charges_disp[:N]
        shell_disp = charges_disp[N+1:].reshape(N, 3)
        return charges, shell_disp

    def compute_shell_disp_qqdd2(self, Vij, Vij_qz, Vij_zq, Vij_zz,
                                 Vij_qz2, Vij_zq2, Vij_qq2, Vij_qq3,
                                 E_d1, E_d2, E_qd, atomic_q0, charges,
                                 field_kernel_e, field_kernel_qe):
        """
        Compute shell displacements through solution of linear system.
        """
        dq = charges - atomic_q0
        nat = dq.shape[0]

        charge_ij = charges[:, None] * charges[None, :]

        # Compute A_ia contributions
        A_ia = 0.5 * jnp.sum((jnp.transpose(Vij_qz, (1, 0, 2)) + Vij_zq) * charges[None, :, None], axis=1)
        A_ia += 0.5 * jnp.sum((jnp.transpose(Vij_qq2, (1, 0, 2)) - Vij_qq2) * charge_ij[:, :, None], axis=1)

        # Compute A_iab contributions
        A_iab = jnp.sum((jnp.transpose(Vij_qz2, (1, 0, 2, 3)) + Vij_zq2) * charges[None, :, None, None], axis=1)
        A_iab += 0.5 * jnp.sum((jnp.transpose(Vij_qq3, (1, 0, 2, 3)) + Vij_qq3) * charge_ij[:, :, None, None], axis=1)

        # Compute A_ijab
        A_ijab = Vij_zz + A_iab * jnp.eye(nat, dtype=Vij.dtype)[:, :, None, None]
        A_ijab += E_d2[:, None, :, None] * jnp.eye(nat, dtype=Vij.dtype)[:, :, None, None] * jnp.eye(3, dtype=Vij.dtype)[None, None, :, :]
        A_ijab -= (jnp.transpose(Vij_qz2, (1, 0, 3, 2)) + Vij_zq2) * charges[None, :, None, None]
        A_ijab -= (jnp.transpose(Vij_zq2, (1, 0, 3, 2)) + Vij_qz2) * charges[:, None, None, None]
        A_ijab -= 2.0 * Vij_qq3 * charge_ij[:, :, None, None]

        # Flatten for linear solve
        A_iajb = jnp.reshape(jnp.transpose(A_ijab, (0, 2, 1, 3)), (nat*3, nat*3))

        # RHS vector
        A_ia += E_d1 + 0.5 * E_qd * dq[:, None] + field_kernel_e + charges[:, None] * field_kernel_qe
        A_ia = A_ia.reshape(-1)

        # Solve linear system
        shell_disp = jnp.reshape(jnp.linalg.solve(A_iajb, -A_ia[:, None]), (nat, 3))
        return shell_disp

    def compute_charges(self, Vij, E1, E2, atomic_q0, total_charge):
        """
        Compute charges through solution of linear system.
        """
        N = E2.shape[0]

        # Pad E2 and construct Aij
        E2_padded = jnp.concatenate([E2, jnp.array([-1.0], dtype=E2.dtype)], axis=0)
        Aij = jnp.concatenate([Vij, jnp.ones((N, 1), dtype=Vij.dtype)], axis=1)
        Aij = jnp.concatenate([Aij, jnp.ones((1, N+1), dtype=Vij.dtype)], axis=0)
        Aij += jnp.diag(E2_padded)

        # Pad E1
        E1_padded = jnp.concatenate([-E1, jnp.array([total_charge], dtype=E1.dtype)], axis=0)

        # Solve linear system
        charges = jnp.linalg.solve(Aij, E1_padded[:, None])

        # Return charges without last padded element
        return jnp.reshape(charges, [-1])[:-1]
    def compute_coulumb_energy(self, charges, atomic_q0, E1, E2, Vij):
        """
        Energy = sum_i E1[i] * dq[i] + 0.5 * (E2[i] * dq[i]^2 + sum_j Vij[i,j] * q[i] * q[j])
        """
        q = charges
        dq = q - atomic_q0
        dq2 = dq * dq
        q_outer = jnp.outer(q, q)  # q_i * q_j
        E = E1 * dq + 0.5 * (E2 * dq2 + jnp.sum(Vij * q_outer, axis=-1))
        return jnp.sum(E)

    def compute_coulumb_energy_pqeq_qd(self, charges, atomic_q0, E1, E2, 
                                       shell_disp, Vij_qq, Vij_qz, Vij_zq, Vij_zz):
        """
        Compute the Coulomb energy with shell displacements (q-delta-PQEq).
        Energy = sum_i E1[i]*q_i + 0.5 * [E2[i]*dq_i^2 + sum_ij Vij_qq q_i q_j 
                 + sum_ij sum_a Vij_qz q_i d_j^a + sum_ij sum_ab Vij_zz d_i^a d_j^b]
        """
        dq = charges - atomic_q0
        dq2 = dq * dq
        q_outer = charges[:, None] * charges[None, :]
        
        # Shell displacement outer product: shape [N,N,3,3]
        shell_disp_outer = shell_disp[:, None, :, None] * shell_disp[None, :, None, :]
        
        # Cross terms q_i d_j^a and q_j d_i^a
        qi_dj = charges[:, None, None] * shell_disp[None, :, :]
        qj_di = charges[None, :, None] * shell_disp[:, None, :]
        E = E1 * dq
        E += 0.5 * (E2 * dq2 + jnp.sum(
            Vij_qq * q_outer +
            jnp.sum(Vij_qz * qi_dj + Vij_zq * qj_di, axis=2) +
            jnp.sum(Vij_zz * shell_disp_outer, axis=(2, 3)),
            axis=-1
        ))
        return jnp.sum(E)


    def compute_coulumb_energy_pqeq(self, charges, atomic_q0, nuclei_charge, 
                                    E1, E2, Vij_qq, Vij_qz, Vij_zq, Vij_zz):
        """
        Compute the Coulomb energy for PQEq without shell displacements.
        Energy = sum_i E1[i]*q_i + 0.5 * [E2[i]*dq_i^2 + sum_ij Vij_qq q_i q_j
                 + sum_ij Vij_qz q_i z_j + sum_ij Vij_zq z_i q_j + sum_ij Vij_zz z_i z_j]
        """
        dq = charges - atomic_q0
        dq2 = dq * dq
        q_outer = charges[:, None] * charges[None, :]
        qz_outer = charges[:, None] * nuclei_charge[None, :]
        zq_outer = charges[None, :] * nuclei_charge[:, None]
        zz_outer = nuclei_charge[:, None] * nuclei_charge[None, :]
        
        E = E1 * dq
        E += 0.5 * (E2 * dq2 + jnp.sum(Vij_qq * q_outer + Vij_qz * qz_outer + 
                                       Vij_zq * zq_outer + Vij_zz * zz_outer, axis=-1))
        return jnp.sum(E)

