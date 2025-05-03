import tensorflow as tf
@tf.function
def unpack_data(data):
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
    result.append(data['energy'])
    result.append(data['forces'])
    return result
