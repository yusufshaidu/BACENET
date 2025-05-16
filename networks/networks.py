import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')
#mixed_precision.set_global_policy('float32')

#@tf.function
def Networks(input_size, layer_sizes,
             activations,
            weight_initializer='glorot_uniform',
            bias_initializer='zeros',
            kernel_constraint=None,
            bias_constraint=None,
            prefix='main',
            l1=0.0,l2=0.0,
            normalize=False):

    model = tf.keras.Sequential(name=f'{prefix}_net')
    model.add(tf.keras.Input(shape=(input_size,)))
    i = 0
    # it seems we now need to specify the input shape differently

    '''
    layer = layer_sizes[0]
    activation = activations[0]
    model.add(tf.keras.layers.Dense(layer,input_shape=(input_size,),
                                        activation=activations[0],
                                        kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        bias_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        activity_regularizer=None, #regularizers.L1L2(l1=l1,l2=l2),
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        trainable=True,
                                        name=f'{prefix}_{i}_layer_{layer}_activation_{activation}'
                                        ))

    i += 1
    '''
    #n_layer = tf.shape(layer_sizes)[0]
    #for i in tf.range(n_layer):
    #    layer = layer_size[i]
    #    activation = activations[i]
    for layer, activation in zip(layer_sizes, activations):
        model.add(tf.keras.layers.Dense(layer,
                                        activation=activation,
                                        kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        bias_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        activity_regularizer=None, #regularizers.L1L2(l1=l1,l2=l2),
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        trainable=True,
 #                                       dtype = mixed_precision.set_global_policy('float32'),
                                        name=f'{prefix}_{i}_layer_{layer}_activation_{activation}'
                                        ))
    #    if normalize and i == 0:
#   #         model.add(tf.keras.layers.BatchNormalization())
    #        model.add(tf.keras.layers.LayerNormalization())
            #model.add(tf.keras.layers.Attention())
        i += 1
    '''
    if activations[-1] == 'linear':
        model.add(tf.keras.layers.Dense(layer_sizes[-1],
                                        kernel_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        kernel_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        bias_regularizer=regularizers.L1L2(l1=l1,l2=l2),
                                        activity_regularizer=None, #regularizers.L1L2(l1=l1,l2=l2),
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        dtype = mixed_precision.set_global_policy('float32'),
                                        trainable=True,
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
                                        activity_regularizer=None, #regularizers.L1L2(l1=l1,l2=l2),
                                        kernel_constraint=kernel_constraint,
                                        bias_constraint=bias_constraint,
                                        dtype = mixed_precision.set_global_policy('float32'),
                                        trainable=True,
                                        name=f'{prefix}_{i}_layer_{layer}_activation_{activations[-1]}'
                                        ))

        #if normalize:
        #    model.add(tf.keras.layers.LayerNormalization())

  #      model.add(tf.keras.layers.BatchNormalization())
      '''
    #model.build((None,input_size))
    return model
