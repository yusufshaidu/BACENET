from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from IPython.display import clear_output
#import tensorflow
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
import math 
import os

import argparse
from data_processing import data_preparation
from model import mBP_model
from model_modified import mBP_model as mBP_model_v1

def create_model(config_file):


    #Read in model parameters
    #I am parsing yaml files with all the parameters
    #
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
    #trainable linear model
    params_trainable = configs['params_trainable']
    pbc = configs['pbc'] 
    if pbc:
        pbc = [True,True,True]
    else:
        pbc = [False,False,False]
    initial_lr = configs['initial_lr']
    model_call = mBP_model
    print('model_version' in list(configs.keys()))
    if 'model_version' in list(configs.keys()):
        model_v = configs['model_version']
        if model_v == 'v1':
            model_call = mBP_model_v1

        print('I am using variable width')
    #this is the global step
    decay_step = configs['decay_step']
    decay_rate = configs['decay_rate']
    #activations are basically sigmoid and linear for now
    activations = ['sigmoid', 'sigmoid', 'linear']
    species = configs['species']
    batch_size = configs['batch_size']
    model_outdir = configs['outdir']
    num_epochs = configs['num_epochs']
    data_dir = configs['data_dir']
    data_format = configs['data_format']
    energy_key = configs['energy_key']
    force_key = configs['force_key']
    test_fraction = configs['test_fraction']
    try:
        l1_norm = configs['l1_norm']
        l2_norm = configs['l2_norm']
    except:
        l1_norm = 0.0
        l2_norm = 0.0

    try:
        atomic_energy = configs['atomic_energy']
    except:
        atomic_energy = []
    
    train_data, test_data, species_identity = data_preparation(data_dir, species, data_format,
                     energy_key, force_key,
                     rc_rad, rc_ang, pbc, batch_size,
                     test_fraction=test_fraction,
                     atomic_energy=atomic_energy)
    
    train_writer = tf.summary.create_file_writer(model_outdir+'/train')

    model = model_call(layer_sizes,
                      rc_rad, species_identity, width, batch_size,
                      activations,
                      rc_ang,RsN_rad,RsN_ang,
                      thetaN,width_ang,zeta,
                      params_trainable,
                      fcost=fcost,
                      pbc=pbc,
                      nelement=nelement,
                      train_writer=train_writer,
                       l1=l1_norm,l2=l2_norm)

    initial_learning_rate = initial_lr

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    

    # Create a callback that saves the model's weights every 5 epochs
    if not os.path.exists(model_outdir):
        os.mkdir(model_outdir)
    checkpoint_path = model_outdir+"/models/ckpts-{epoch:04d}.ckpt"

    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                    save_freq=save_freq,
                                                    options=None),
                   tf.keras.callbacks.TensorBoard(model_outdir, histogram_freq=1,
                                                  update_freq='batch'),
                   tf.keras.callbacks.BackupAndRestore(backup_dir=model_outdir+"/tmp_backup"),
                   tf.keras.callbacks.CSVLogger(model_outdir+"/metrics.dat", separator=" ", append=True)]


    model.save_weights(checkpoint_path.format(epoch=0))
    #train the model
    
        
    #load the last saved epoch

    model.compile(optimizer=optimizer, loss="mse", metrics=["MAE", 'loss'])
 #   try:
 #       model.fit(train_data,
 #            epochs=num_epochs,
 #            batch_size=batch_size,
 #            validation_data=test_data,
 #            validation_freq=10,
 #            callbacks=[cp_callback])
 #   except:
#      pass

    model.fit(train_data,
              epochs=num_epochs,
              batch_size=batch_size,
             validation_data=test_data,
             validation_freq=5,
             callbacks=[cp_callback])

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
    



