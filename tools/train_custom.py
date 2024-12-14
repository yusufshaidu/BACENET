from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
#from IPython.display import clear_output
#import tensorflow
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
from swa.tfkeras import SWA
import math 
import os, json
from tensorflow.keras.optimizers import Adam
import helping_functions as help_fn




import argparse
from data_processing import data_preparation
#from model import mBP_model
#from model_legendre_polynomial import mBP_model
from model_modified_manybody import mBP_model
from model_modified_manybody_linear_scaling import mBP_model as mBP_model_linear

#from tensorflow.keras import mixed_precision
#mixed_precision.set_global_policy('mixed_float16')

def compute_features(train_data, model):
    inputs_target = train_data
    inputs = inputs_target[:6]
    target = tf.cast(inputs_target[6], tf.float32)

    batch_nats = tf.cast(inputs[2], tf.float32)
    nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)

    target_f = tf.reshape(inputs_target[7].to_tensor(), [-1, 3*nmax])
    target_f = tf.cast(target_f, tf.float32)

    features, feature_size = model.call(inputs, training=False)
    return features, feature_size

#@tf.function
def custom_train_step(train_data, model, ecost, fcost):
    inputs_target = train_data
    inputs = inputs_target[:6]
    target = tf.cast(inputs_target[6], tf.float32)

    batch_nats = tf.cast(inputs[2], tf.float32)
    nmax = tf.cast(tf.reduce_max(batch_nats), tf.int32)

    target_f = tf.reshape(inputs_target[7].to_tensor(), [-1, 3*nmax])
    target_f = tf.cast(target_f, tf.float32)

    with tf.GradientTape() as tape:
        e_pred, forces, atomic_features = model.call(inputs, training=False)  # Forward pass
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        ediff = (e_pred - target)
        forces = tf.reshape(forces, [-1, 3*nmax])

        #emse_loss = tf.reduce_mean((ediff/batch_nats)**2)
        emse_loss = tf.reduce_mean((ediff)**2)

        fmse_loss = tf.map_fn(help_fn.force_loss, (batch_nats,target_f,forces), fn_output_signature=tf.float32)
        fmse_loss = tf.reduce_mean(fmse_loss)

        loss = ecost * emse_loss
        loss += fcost * fmse_loss
    trainable_vars = model.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    return model, gradients, loss, tf.sqrt(emse_loss), tf.sqrt(fmse_loss), atomic_features


def default_config():
    configs = {}
    configs['layer_sizes'] = None
    configs['nelement'] = 118
    configs['save_freq'] = 'epoch'
    configs['zeta'] =  50
    configs['thetaN'] = None
    configs['RsN_rad'] = None
    configs['RsN_ang'] = None
    configs['rc_rad'] = None
    configs['rc_ang'] = None
    configs['fcost'] = None
    configs['ecost'] = 1.0
    #trainable linear model
    configs['pbc'] = True
    configs['initial_lr'] = 0.001
    configs['model_version'] = 'linear'
    #this is the global step
    configs['decay_step'] = None
    configs['decay_rate'] = None
    #activations are basically tanh and linear for now
    configs['species'] = None
    configs['batch_size'] = 4
    configs['outdir'] = './train'
    configs['num_epochs'] = 10000000
    configs['data_dir'] = 'jsons'
    configs['data_format'] = 'panna_json'
    configs['energy_key'] = 'energy'
    configs['force_key'] = 'forces'
    configs['test_fraction'] = 0.1
    configs['nspec_embedding'] = 4
    
    configs['l1_norm'] = 0.0
    configs['l2_norm'] = 0.0
    configs['include_vdw'] = False
    configs['atomic_energy'] = []
    configs['rmin_u'] = 3.0
    configs['rmax_u'] = 5.0
    configs['rmin_d'] = 10.0
    configs['rmax_d'] = 12.0
    configs['opt_method'] = 'adam'
    configs['fixed_lr'] = False
    configs['body_order'] = 3
    configs['min_radial_center'] = 0.5
    configs['species_out_act'] = 'linear'
    configs['start_swa'] = -1
    configs['min_lr'] = 0.0001
    configs['swa_lr'] = 0.0001
    configs['swa_lr2'] = 0.001
    return configs

def create_model(configs):


    #Read in model parameters
    #I am parsing yaml files with all the parameters
    #
    _configs = default_config()
    
    for key in _configs:
        if key not in configs.keys():
            configs[key] = _configs[key]
    layer_sizes = configs['layer_sizes']
    nelement = configs['nelement']
    save_freq = configs['save_freq']
    zeta = configs['zeta']
    thetaN = configs['thetaN']
    RsN_rad = configs['RsN_rad']
    RsN_ang = configs['RsN_ang']
    rc_rad = configs['rc_rad'] 
    rc_ang = configs['rc_ang'] 
    #estimate initial parameters
    width_ang = RsN_ang * RsN_ang / (rc_ang-0.25)**2
    width = RsN_rad * RsN_rad / (rc_rad-0.25)**2
    fcost = configs['fcost']
    ecost = configs['ecost']
    #trainable linear model
    _pbc = configs['pbc'] 
    if _pbc:
        pbc = [True,True,True]
    else:
        pbc = [False,False,False]
    initial_lr = configs['initial_lr']
    model_call = mBP_model
    print('model_version' in list(configs.keys()))
    if 'model_version' in list(configs.keys()):
        model_v = configs['model_version']
        if model_v == 'linear':
            model_call = mBP_model_linear

        print('I am using variable width')
    #this is the global step
    decay_step = configs['decay_step']
    decay_rate = configs['decay_rate']
    #activations are basically tanh and linear for now
    activations = configs['activations']
    print(activations)
    
    assert len(activations) == len(layer_sizes),'the number of activations must be same as the number of layer'
    if activations[-1] != 'linear':
        print(f'You have set the last layer to {activations[-1]} but must be st to linear')
        print(f'we set it to linear')
        activations[-1] = 'linear'
    
    species = configs['species']
    batch_size = configs['batch_size']
    model_outdir = configs['outdir']
    if not os.path.exists(model_outdir):
        os.mkdir(model_outdir)

    num_epochs = configs['num_epochs']
    data_dir = configs['data_dir']
    data_format = configs['data_format']
    energy_key = configs['energy_key']
    force_key = configs['force_key']
    test_fraction = configs['test_fraction']
    l1_norm = configs['l1_norm']
    l2_norm = configs['l2_norm']
    l1_norm = 0.0
    l2_norm = 0.0

    atomic_energy = configs['atomic_energy']
    include_vdw = configs['include_vdw']
    rmin_u = configs['rmin_u']
    rmax_u = configs['rmax_u']
    rmin_d = configs['rmin_d']
    rmax_d = configs['rmax_d']
    opt_method = configs['opt_method']
    fixed_lr = configs['fixed_lr']
    body_order = configs['body_order']
    min_radial_center = configs['min_radial_center']
    species_out_act = configs['species_out_act']
    start_swa = configs['start_swa']
    swa_lr = configs['swa_lr']
    min_lr = configs['min_lr']
    swa_lr2 = configs['swa_lr2']

    if include_vdw:
        rc = np.max([rc_rad,rc_ang,rmax_d])
    else:
        rc = np.max([rc_rad,rc_ang])
    train_data, test_data, species_identity = data_preparation(data_dir, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction=test_fraction,
                     atomic_energy=atomic_energy, 
                     atomic_energy_file=os.path.join(model_outdir,'atomic_energy.json'))
    initial_learning_rate = initial_lr
    #if fixed_lr:
    #    lr_schedule = initial_learning_rate
    #else:
    #    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #        initial_learning_rate,
    #        decay_steps=decay_step,
    #        decay_rate=decay_rate,
    #        staircase=True)
    lr_schedule = initial_learning_rate

    cb_ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='RMSE_F',
        factor=decay_rate,
        patience=decay_step,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=min_lr,
        )

    #callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='RMSE_F',
    #                                          patience=decay_step,
    #                                          baseline = 1e-5
   #                                             )
    if opt_method in ['adamW', 'AdamW', 'adamw']:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.0001,
            amsgrad=False,
            clipnorm=None,
            clipvalue=None,
            use_ema=False,
            name='adamw')

        print('using adamW as the optimizer')
    elif opt_method in ['Adadelta', 'adadelta']:
        optimizer = tf.keras.optimizers.Adadelta(
            learning_rate=lr_schedule,
            weight_decay=0.0001,
            clipnorm=None,
            clipvalue=None,
            use_ema=False,
            name='adadelta')
        print('using adadelta as the optimizer')
    else:
        print('using adam as the optimizer')
        optimizer = tf.keras.optimizers.Adam(
                                             learning_rate=lr_schedule,
                                             use_ema=False,
                                             weight_decay=0.0, 
                                             clipnorm=None,
                                             clipvalue=None,
                                             amsgrad=False,
                                             )
    
#    train_writer = tf.summary.create_file_writer(model_outdir+'/train')
    
    if not os.path.exists(model_outdir):
        os.mkdir(model_outdir)
    #checkpoint_path = model_outdir+"/models/ckpts-{epoch:04d}-{val_loss:.5f}.keras"
#    checkpoint_path = model_outdir+"/models/ckpts-{epoch:04d}.keras"
    checkpoint_path = model_outdir+"/models/ckpts-{epoch:04d}.ckpt"

    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    if start_swa > -1:

        swa = SWA(start_epoch=start_swa, 
              lr_schedule='manual', # other options are constant and cyclic (cycle between lr1 and lr2)
              swa_lr=swa_lr, 
              swa_lr2=swa_lr2,
              swa_freq=5,
              verbose=1)


        cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                    save_freq=save_freq),
                   tf.keras.callbacks.TensorBoard(model_outdir, histogram_freq=1,
                                                  update_freq='epoch'),
                   tf.keras.callbacks.CSVLogger(model_outdir+"/metrics.dat", separator=" ", append=True),
                       cb_ReduceLROnPlateau]
    else:
        cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                    save_freq=save_freq),
                   tf.keras.callbacks.TensorBoard(model_outdir, histogram_freq=1,
                                                  update_freq='epoch'),
                   tf.keras.callbacks.CSVLogger(model_outdir+"/metrics.dat", separator=" ", append=True),
                       cb_ReduceLROnPlateau]



    backupandrestore = tf.keras.callbacks.BackupAndRestore(backup_dir=model_outdir+"/tmp_backup", delete_checkpoint=False)
    cp_callback.append(backupandrestore)

    
    '''
    try:
        # This should work well on multiple GPUs on a single computer
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():

            model = model_call(layer_sizes,
                      rc_rad, species_identity, width, batch_size,
                      activations,
                      rc_ang,RsN_rad,RsN_ang,
                      thetaN,width_ang,zeta,
                      fcost=fcost,
                      ecost=ecost,
                      pbc=_pbc,
                      nelement=nelement,
                      nspec_embedding=configs['nspec_embedding'],
                      train_writer=model_outdir,
                      l1=l1_norm,l2=l2_norm,
                      include_vdw=include_vdw,
                      rmin_u=rmin_u,rmax_u=rmax_u,
                      rmin_d=rmin_d,rmax_d=rmax_d,
                      body_order=body_order,
                      min_radial_center=min_radial_center,
                      species_out_act=species_out_act)
            #train the model
            model.compile(optimizer=optimizer, loss="mse", metrics=["MAE", 'loss'])
    except:
    '''
    #preprocessing
    model = model_call(layer_sizes,
                      rc_rad, species_identity, width, batch_size,
                      activations,
                      rc_ang,RsN_rad,RsN_ang,
                      thetaN,width_ang,zeta,
                      fcost=fcost,
                      ecost=ecost,
                      pbc=_pbc,
                      nelement=nelement,
                      nspec_embedding=configs['nspec_embedding'],
                      train_writer=model_outdir,
                      l1=l1_norm,l2=l2_norm,
                      include_vdw=include_vdw,
                      rmin_u=rmin_u,rmax_u=rmax_u,
                      rmin_d=rmin_d,rmax_d=rmax_d,
                      body_order=body_order,
                      min_radial_center=min_radial_center,
                      species_out_act=species_out_act, 
                      features=True)
    i = 0
    nconf = 0
    for _train_data in train_data:
        features, feature_size = compute_features(_train_data, model)
        features = tf.reshape(features, [-1, feature_size])
        fshape = tf.shape(features)
        nconf += tf.get_static_value(fshape[0])

        if i == 0:
            all_features = tf.zeros(feature_size)
            all_features2 = tf.zeros(feature_size)
            print(feature_size, nconf, fshape[0])

        all_features += tf.reduce_sum(features, axis=0)
        all_features2 += tf.reduce_sum(features*features, axis=0)
        i += 1
    mean_features = all_features / nconf 
    std_features = tf.sqrt(all_features2 / nconf - mean_features*mean_features)
    std_features = tf.ones(feature_size)

    print(mean_features,std_features)

     # It should work on CPU platform
    model = model_call(layer_sizes,
                      rc_rad, species_identity, width, batch_size,
                      activations,
                      rc_ang,RsN_rad,RsN_ang,
                      thetaN,width_ang,zeta,
                      fcost=fcost,
                      ecost=ecost,
                      pbc=_pbc,
                      nelement=nelement,
                      nspec_embedding=configs['nspec_embedding'],
                      train_writer=model_outdir,
                      l1=l1_norm,l2=l2_norm,
                      include_vdw=include_vdw,
                      rmin_u=rmin_u,rmax_u=rmax_u,
                      rmin_d=rmin_d,rmax_d=rmax_d,
                      body_order=body_order,
                      min_radial_center=min_radial_center,
                      species_out_act=species_out_act,
                      mean_descriptors=mean_features,
                      std_descriptors=std_features)

    callbacks = tf.keras.callbacks.CallbackList(
    cp_callback, add_history=True, model=model)

    logs = {}
    callbacks.on_train_begin(logs=logs)

    # Keep results for plotting
    #train_loss_results = []
    #train_accuracy_results = []

    #num_epochs = 201
    global_step = 0
    for epoch in range(num_epochs):
        callbacks.on_epoch_begin(epoch, logs=logs)

        # Training loop - using batches of 32
        for i, _train_data in enumerate(train_data):
            model.reset_states()
            callbacks.on_batch_begin(i, logs=logs)
            callbacks.on_train_batch_begin(i, logs=logs)
            model, gradients, loss, emse_loss, fmse_loss,_ = custom_train_step(_train_data, model, ecost,fcost)
            trainable_vars = model.trainable_variables
            # Update weights
            optimizer.apply_gradients(zip(gradients, trainable_vars))
             
            if global_step % 10 == 0:
                print("Epoch {}: step: {} E_MAE: {:.3f}, F_MAE: {:.3f}".format(epoch,global_step,
                                                                        emse_loss,
                                                                        fmse_loss))
            global_step += 1
        model.save_weights(checkpoint_path.format(epoch=epoch))


    '''
    model.compile(optimizer=optimizer, loss="mse", metrics=["MAE", 'loss'])
    model.save_weights(checkpoint_path.format(epoch=0))
    try:
        model.fit(train_data,
             epochs=num_epochs,
             batch_size=batch_size,
             validation_data=test_data,
             validation_freq=10,
             callbacks=[cp_callback,backupandrestore])
    except:
#      pass

        model.fit(train_data,
              epochs=num_epochs,
              batch_size=batch_size,
             validation_data=test_data,
             validation_freq=10,
             callbacks=[cp_callback])
    '''
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='create ML model')
    parser.add_argument('-c', '--config', type=str,
                        help='configuration file', required=True)
    args = parser.parse_args()


    import yaml
    with open(args.config) as f:
        configs = yaml.safe_load(f)

    print(configs)

#    configs = {}
#    num_epochs = configs['num_epochs']
#    batch_size = configs['batch_size']
#    data_dir = configs['data_dir']
#    data_format = configs['data_format']
#    species = configs['species']
#    energy_key = configs['energy_key']
#    force_key = configs['force_key']
#    test_fraction = configs['force_key']
#    layer_sizes = configs['layer_sizes']
#    save_freq = configs['save_freq']
#    zeta = configs['zeta']
#    thetaN = configs['thetaN']
#    RsN_rad = configs['RsN_rad']
#    RsN_ang = configs['RsN_ang']
#    rc_rad = configs['rc_rad']
#    rc_ang = configs['rc_ang']
#    fcost = configs['fcost']
    #trainable linear model
#    params_trainable = configs['params_trainable']
#    pbc = configs['pbc']
#    initial_lr = configs['lr']
#    #this is the global step
#    decay_step = configs['decay_step']
#    decay_rate = configs['decay_rate']
#    model_outdir = configs['outdir']


    create_model(configs)
    



