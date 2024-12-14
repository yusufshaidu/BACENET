###########################################################################
# Copyright (c), The PANNAdevs group. All rights reserved.                #
# This file is part of the PANNA code.                                    #
#                                                                         #
# The code is hosted on GitLab at https://gitlab.com/PANNAdevs/panna      #
# For further information on the license, see the LICENSE.txt file        #
###########################################################################
import numpy as np
import itertools

def orthogonal_vector(input_vector):
    """ calculate a random orthogonal unit vector to input_vector
        by solving the equation ax * x + ay * y + az * z = 0
        with input_vector = [ax, ay, az]
        candidate_vector = [x, y, z]

    Parameters
    ----------
    input_vector : 1D array
        a vector

    Returns
    -------
    1D array
        candidate_vector, random unit vector
    """
    candidate_vector = np.zeros(3)
    count = 0  # check for first non-zero element of input_vector
    for idx in range(3):

        if input_vector[idx] != 0 and count < 1:
            # choose only one of the components of candidate vector and the
            # other one is zero (this is sufficient !)
            candidate_vector[(idx + 1) % 3] = np.random.random(1)
            # candidate_vector[(idx + 2) % 3] = np.random.random(1)

            candidate_vector[idx] = -(candidate_vector[(idx + 1) % 3]
                                      * input_vector[(idx + 1) % 3]
                                      + candidate_vector[(idx + 2) % 3]
                                      * input_vector[(idx + 2) % 3]) \
                / input_vector[idx]
            count += 1

    candidate_vector = candidate_vector / np.linalg.norm(candidate_vector)

    return candidate_vector


def replicas_max_idx(lattice_vectors, Rc, pbc=[True, True, True]):
    """ calculate maximum number of cells needed with a given radial cutoff

    Parameters
    ----------
    lattice_vectors: lattice vectors as matrix (a1, a2, a3)
    Rc: maximum radial cutoff that you want to take in to account
    pbc: Normally pbc are recovered from lattice vector,
         if in the lattice_vectors a direction is set to zero
         then no pbc is applied in that direction.
         This argument allow you to turn off specific directions
         in the case where that specific direction has a lattice vector
         greater then zero.
         To achieve this pass an array of 3 logical value (one for each
         direction). False value turn off that specific direction.
         Default is true in every direction.
         eg. pbc = [True, False, True] => pbc along a1 and a3

    Returns
    -------
    max_indices: [lmax, mmax, nmax], numpy array
       integers for the number of replicas.
    """
    if not isinstance(lattice_vectors, np.ndarray):
        _lattice_vectors = np.asarray(lattice_vectors)
    else:
        _lattice_vectors = lattice_vectors.copy()

    lattice_vector_lengths = np.linalg.norm(_lattice_vectors, axis=1)
    lattice_vector_lengths_bool = lattice_vector_lengths > 1e-6
    max_indices = np.zeros(3, dtype=int)
    if not lattice_vector_lengths_bool.any():
        return max_indices

    lattice_vectors_idxs = np.where(lattice_vector_lengths_bool)[0]
    lattice_vectors_idxs_false = np.where(~lattice_vector_lengths_bool)[0]

    # define an index control
    lat_vec_idx_control = len(lattice_vectors_idxs)

    for idx in lattice_vectors_idxs_false:

        # when only one vector is has a nonzero length
        if lat_vec_idx_control == 1:

            j_idx = lattice_vectors_idxs[0]
            # orthogonal vector to _lattice_vectors[j_idx
            _lattice_vectors[idx] = orthogonal_vector(_lattice_vectors[j_idx])

        # when two vectors have nonzero length/given
        # compute the  cross_product  of the two given vectors

        if lat_vec_idx_control == 2:
            _lattice_vectors[idx] = np.cross(
                _lattice_vectors[(idx - lat_vec_idx_control) % 3],
                _lattice_vectors[(idx - lat_vec_idx_control + 1) % 3])
            _lattice_vectors[idx] = _lattice_vectors[idx] \
                / np.linalg.norm(_lattice_vectors[idx])
        lat_vec_idx_control += 1

    reciprocal_vectors = np.linalg.inv(_lattice_vectors)
    reciprocal_vectors_length = np.linalg.norm(reciprocal_vectors, axis=0)

    for idx in lattice_vectors_idxs:
        if pbc[idx]:
            b_length = reciprocal_vectors_length[idx]
            max_indices[idx] = int(np.ceil(Rc * b_length))

    return max_indices
