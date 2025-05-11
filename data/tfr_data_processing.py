import tensorflow as tf
from functools import partial
import numpy as np
import os
from data.unpack_tfr_data import np_unpack_data
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    #return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def example(data,nmax, nneigh_max):
    # Create a description of the features.
    nat = int(data[4])
    ndiff = (nmax - nat)
    positions = np.pad(data[0].flatten(), pad_width=(0,ndiff*3)).tolist()
    forces = np.pad(data[11].flatten(), pad_width=(0,ndiff*3)).tolist()
    atomic_number = np.pad(data[1].flatten(), pad_width=(0,ndiff)).tolist()
    C6 = np.pad(data[2].flatten(), pad_width=(0,ndiff)).tolist()
    gaussian_width = np.pad(data[9].flatten(), pad_width=(0,ndiff)).tolist()
    nneigh = int(data[8])
    ndiff = nneigh_max - nneigh
    i_idx = np.pad(data[5].flatten(), pad_width=(0,ndiff)).tolist()
    j_idx = np.pad(data[6].flatten(), pad_width=(0,ndiff)).tolist()
    S_idx = np.pad(data[7].flatten(), pad_width=(0,ndiff*3)).tolist()
    features = {'positions':_float_feature(positions),
                'atomic_number':_float_feature(atomic_number),
                'C6':_float_feature(C6),
                'cells':_float_feature(data[3].flatten().tolist()),
                'natoms':_int64_feature([data[4]]),
                'i':_int64_feature(i_idx),
                'j':_int64_feature(j_idx),
                'S':_int64_feature(S_idx),
                'nneigh':_int64_feature([data[8]]),
                'gaussian_width':_float_feature(gaussian_width),
                'energy':_float_feature([data[10]]),
                'forces':_float_feature(forces)}
    return tf.train.Example(features=tf.train.Features(feature=features))

def write_tfr(prefix, data, nmax, nneigh_max, nelement=None, tfr_dir='tfrs',):
    #data contains [positions,species_encoder,C6,cells,natoms,i,j,S,neigh, energy,forces]
    #each has nconf as the first axis dimension
    if not os.path.exists(tfr_dir):
        os.mkdir(tfr_dir)
    
    _data = np_unpack_data(data)

    num_examples = len(_data[0])
    n_items = len(_data)
    if nelement is None:
        nelement = num_examples 
    nfiles = num_examples // nelement
    if nelement * nfiles < num_examples:
        nfiles += 1
    #print(f'number of tfrs: {nfiles}, number of examples: {num_examples}')
    for n in range(nfiles):
        #if n % nfiles == 0:
        filename = tfr_dir+f'/{prefix}_{n}.tfrecords'
        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(n*nelement, (n+1)*nelement):
                if i >= num_examples:
                    break
                inputs = [_data[j][i] for j in range(n_items)]
                tf_example = example(inputs, nmax, nneigh_max)
                writer.write(tf_example.SerializeToString())
                #writer.write(tf_example)
        writer.close()

def _parse_function(example_proto):
    # Parse the input tf.train.Example proto using the dictionary above.

    feature_description = {'positions':tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True),
                'atomic_number':tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True),
                'C6':tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True),
                'cells':tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True),
                'natoms':tf.io.FixedLenFeature([], tf.int64),
                'i':tf.io.FixedLenSequenceFeature([],tf.int64,allow_missing=True),
                'j':tf.io.FixedLenSequenceFeature([],tf.int64,allow_missing=True),
                'S':tf.io.FixedLenSequenceFeature([],tf.int64,allow_missing=True),
                'nneigh':tf.io.FixedLenFeature([], tf.int64),
                'gaussian_width':tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True),
                'energy':tf.io.FixedLenFeature([], tf.float32),
                'forces':tf.io.FixedLenSequenceFeature([],tf.float32, allow_missing=True)}
    return tf.io.parse_single_example(example_proto, feature_description)

def load_tfrs(filenames):
    AUTOTUNE = tf.data.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames,
        #buffer_size=16,
        num_parallel_reads=16
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(_parse_function, 
                          num_parallel_calls=AUTOTUNE)
    # returns a dataset
    #dataset.cache()
    return dataset

def get_tfrs(filenames, batch_size):
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = load_tfrs(filenames)
    dataset = dataset.shuffle(1204)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size,
                            num_parallel_calls=AUTOTUNE,
                            deterministic=False)
    return dataset
