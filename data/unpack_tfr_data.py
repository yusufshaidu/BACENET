import tensorflow as tf

@tf.function(jit_compile=False
             )
def unpack_data(data):
    return (
        data['positions'],
        data['atomic_number'],
        data['C6'],
        data['cells'],
        data['natoms'],
        data['i'],
        data['j'],
        data['S'],
        data['nneigh'],
        data['gaussian_width'],
        data['energy'],
        data['forces'],
        data['total_charge'],
        data['charges'],
        data['stress']
    )
def np_unpack_data(data):
    #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh, energy,forces]
    return (
        data['positions'],
        data['atomic_number'],
        data['C6'],
        data['cells'],
        data['natoms'],
        data['i'],
        data['j'],
        data['S'],
        data['nneigh'],
        data['gaussian_width'],
        data['energy'],
        data['forces'],
        data['total_charge'],
        data['charges'],
        data['stress']
    )
