import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
#mixed_precision.set_global_policy('float32')


def Networks(input_size, layer_sizes,
             activations,
            weight_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_constraint=None,
            bias_constraint=None,
            prefix='main',
            l1=0.0,l2=0.0,
            normalize=False):

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    i = 0
    layer = 1
    for layer, activation in zip(layer_sizes[:-1], activations[:-1]):
        model.add(tf.keras.layers.Dense(layer,
                                        activation=activation,
                                        kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        bias_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        activity_regularizer=None,
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        trainable=True,
                                        dtype=mixed_precision.set_global_policy('float32'),
                                        name=f'{prefix}_{i}_layer_{layer}_activation_{activation}'
                                        ))
#        model.add(tf.keras.layers.BatchNormalization())
        if normalize:
            model.add(tf.keras.layers.LayerNormalization())
        i += 1

    if activations[-1] == 'linear':
        model.add(tf.keras.layers.Dense(layer_sizes[-1],
                                        kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        bias_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        activity_regularizer=None,
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        trainable=True,
                                        dtype=tf.float32,
                                        name=f'{prefix}_{i}_layer_{layer}_activation_{activations[-1]}'
                                        ))
        #if normalize:
        #    model.add(tf.keras.layers.LayerNormalization())

 #       model.add(tf.keras.layers.BatchNormalization())
    else:
        model.add(tf.keras.layers.Dense(layer_sizes[-1], activation=activations[-1],
                                        kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        bias_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        activity_regularizer=None,
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        trainable=True,
                                        dtype=tf.float32,
                                        name=f'{prefix}_{i}_layer_{layer}_activation_{activations[-1]}'
                                        ))

        #if normalize:
        #    model.add(tf.keras.layers.LayerNormalization())

  #      model.add(tf.keras.layers.BatchNormalization())
    return model
