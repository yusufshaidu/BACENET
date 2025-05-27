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
        data['forces']
    )


def np_unpack_data(data):
    #[positions,species_encoder,C6,cells,natoms,i,j,S,neigh, energy,forces]
    result = []
    result.append(data['positions'])
    result.append(data['atomic_number'])
    result.append(data['C6'])
    result.append(data['cells'])
    result.append(data['natoms'])
    result.append(data['i'])
    result.append(data['j'])
    result.append(data['S'])
    result.append(data['nneigh'])
    result.append(data['gaussian_width'])
    result.append(data['energy'])
    result.append(data['forces'])
    return result
