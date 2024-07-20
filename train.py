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
#from model import mBP_model
#from model_legendre_polynomial import mBP_model
from model_modified_zchannel import mBP_model
from model_modified import mBP_model as mBP_model_v1

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
    configs['params_trainable'] = False
    configs['pbc'] = True
    configs['initial_lr'] = 0.001
    configs['model_version'] = 'v1'
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
    configs['nspec_embedding'] = 64
    
    configs['l1_norm'] = 0.0
    configs['l2_norm'] = 0.0
    configs['train_zeta'] = True
    configs['include_vdw'] = False
    configs['atomic_energy'] = []
    configs['rmin_u'] = 3.0
    configs['rmax_u'] = 5.0
    configs['rmin_d'] = 10.0
    configs['rmax_d'] = 12.0
    configs['lp_lmax'] = 16
    configs['Nzeta'] = None
    configs['dcenters'] = 1.0
    configs['variable_width'] = False
    configs['opt_method'] = 'adam'
    configs['fixed_lr'] = False
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
    Nzeta = configs['Nzeta']
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
    params_trainable = configs['params_trainable']
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
        if model_v == 'v1':
            model_call = mBP_model_v1

        print('I am using variable width')
    #this is the global step
    decay_step = configs['decay_step']
    decay_rate = configs['decay_rate']
    #activations are basically tanh and linear for now
    activations = ['tanh', 'tanh', 'linear']
    species = configs['species']
    batch_size = configs['batch_size']
    model_outdir = configs['outdir']
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

    train_zeta = configs['train_zeta']
    atomic_energy = configs['atomic_energy']
    include_vdw = configs['include_vdw']
    rmin_u = configs['rmin_u']
    rmax_u = configs['rmax_u']
    rmin_d = configs['rmin_d']
    rmax_d = configs['rmax_d']
    lp_lmax = configs['lp_lmax']
    dcenters = configs['dcenters']
    variable_width = configs['variable_width']
    opt_method = configs['opt_method']
    fixed_lr = configs['fixed_lr']

    if include_vdw:
        rc = np.max([rc_rad,rc_ang,rmax_d])
    else:
        rc = np.max([rc_rad,rc_ang])
    train_data, test_data, species_identity = data_preparation(data_dir, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
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
                      ecost=ecost,
                      pbc=_pbc,
                      nelement=nelement,
                      train_writer=train_writer,
                      train_zeta=train_zeta,
                      l1=l1_norm,l2=l2_norm,
                      include_vdw=include_vdw,
                      rmin_u=rmin_u,rmax_u=rmax_u,
                      rmin_d=rmin_d,rmax_d=rmax_d,
                      Nzeta=Nzeta, 
                      dcenters=dcenters,
                      variable_width=variable_width)

    initial_learning_rate = initial_lr
    if fixed_lr:
        lr_schedule = initial_learning_rate
    else:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=decay_step,
            decay_rate=decay_rate,
            staircase=True)
    if opt_method in ['adamW', 'AdamW', 'adamw']:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.004,
            amsgrad=True,
            clipnorm=None,
            clipvalue=None,
            use_ema=True,
            name='adamw')

        print('using adamW as the optimizer')
    elif opt_method in ['Adadelta', 'adadelta']:
        optimizer = tf.keras.optimizers.Adadelta(
            learning_rate=lr_schedule,
            weight_decay=0.004,
            clipnorm=None,
            clipvalue=None,
            use_ema=True,
            name='adadelta')
        print('using adadelta as the optimizer')
    else:
        print('using adam as the optimizer')
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, 
                                             use_ema=True,
                                             weight_decay=0.004, 
                                             clipnorm=None,
                                             clipvalue=None,
                                             )
    

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
                   tf.keras.callbacks.BackupAndRestore(backup_dir=model_outdir+"/tmp_backup", delete_checkpoint=False),
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
    



