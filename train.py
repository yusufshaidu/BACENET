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




import argparse
from data_processing import data_preparation
from model import mBP_model

def default_config():
    return {
        'layer_sizes': None,
        'nelement': 118,
        'save_freq': 'epoch',
        'zeta': 4,
        'rc_rad': 6,
        'Nrad': 8,
        'fcost': 0.0,
        'ecost': 1.0,
        'pbc': True,
        'initial_lr': 0.001,
        'model_version': 'linear',
        'decay_step': None,
        'decay_rate': None,
        'species': None,
        'batch_size': 4,
        'outdir': './train',
        'num_epochs': 10000000,
        'data_dir': 'jsons',
        'data_format': 'panna_json',
        'energy_key': 'energy',
        'force_key': 'forces',
        'test_fraction': 0.1,
        'nspec_embedding': 4,
        'l1_norm': 0.0,
        'l2_norm': 0.0,
        'include_vdw': False,
        'atomic_energy': [],
        'rmin_u': 3.0,
        'rmax_u': 5.0,
        'rmin_d': 10.0,
        'rmax_d': 12.0,
        'opt_method': 'adam',
        'fixed_lr': False,
        'body_order': 3,
        'species_out_act': 'linear',
        'start_swa': -1,
        'min_lr': 1e-5,
        'swa_lr': 1e-4,
        'swa_lr2': 1e-3,
        'clip_value': 1e9,
        'species_layer_sizes': [],
        'species_correlation': 'dot',
        'radial_layer_sizes': [64, 64],
        'learn_radial': True,
        'activations': None,
        'coulumb': False,
        'accuracy': 1e-6,
        'total_charge': 0.0,
        'features': False,
        'normalize': False
    }
def get_compiled_model(configs,optimizer):

    model = mBP_model(configs)
    model.compile(optimizer=optimizer,
                  #loss="mse",
                  metrics=["MAE"])
    return model
#This is an example from tensorflow
#TODO
def make_or_restore_model(model_outdir,configs,optimizer):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoint_dir = model_outdir + "/models/"
    checkpoints = [checkpoint_dir + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint,custom_objects=None)
    print("Creating a new model")
    return get_compiled_model(configs,optimizer)
def create_model(configs):


    #Read in model parameters
    #I am parsing yaml files with all the parameters
    #
    _configs = default_config()
    
    for key in _configs:
        if key not in configs.keys():
            configs[key] = _configs[key]
    model_outdir = configs['outdir']
    if not os.path.exists(model_outdir):
        os.mkdir(model_outdir)

    # Learning rate schedule
    if not configs['fixed_lr']:
        '''cb_ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='RMSE_F',
        factor=decay_rate,
        patience=decay_step,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=min_lr,
        )
        #cp_callback.append(cb_ReduceLROnPlateau)
        '''
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=configs['initial_lr'],
        decay_steps=configs['decay_step'],
        decay_rate=configs['decay_rate'],
        staircase=True,
        name='ExponentialDecay')

    else:
        lr_schedule = configs['initial_lr']
    
    #optimization
    if configs['opt_method'] in ['adamW', 'AdamW', 'adamw']:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-8,
            amsgrad=True,
            clipnorm=None,
            clipvalue=configs['clip_value'],
            use_ema=False,
            name='adamw')

        print('using adamW as the optimizer')
    elif configs['opt_method'] in ['Adadelta', 'adadelta']:
        optimizer = tf.keras.optimizers.Adadelta(
            learning_rate=lr_schedule,
            weight_decay=0.0001,
            clipnorm=None,
            clipvalue=configs['clip_value'],
            use_ema=False,
            name='adadelta')
        print('using adadelta as the optimizer')
    else:
        print('using adam as the optimizer')
        optimizer = tf.keras.optimizers.Adam(
             learning_rate=lr_schedule,
             use_ema=False,
             weight_decay=1e-8, 
             clipnorm=None,
             clipvalue=configs['clip_value'],
             amsgrad=False,
             name='adam')

    #save checkpoints
    checkpoint_dir = model_outdir+"/models"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
   
    checkpoint_path = checkpoint_dir+"/ckpts-{epoch:04d}.ckpt"

    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                    save_freq=configs['save_freq']),
                   tf.keras.callbacks.TensorBoard(model_outdir, histogram_freq=1,
                                                  update_freq='epoch'),
                   tf.keras.callbacks.CSVLogger(model_outdir+"/metrics.dat", separator=" ", append=True)]

    if configs['start_swa'] > -1:
        try:
            epochs = np.loadtxt(model_outdir+"/metrics.dat",skiprows=1, usecols=0)
            last_epoch = int(epochs[-1])
        except:
            last_epoch = 0
        if configs['start_swa'] < last_epoch:
            print(f"start_epoch_swa must be larger than the last saved epoch but are {configs['start_swa']} and {last_epoch}")
            print(f"setting start_swa to {last_epoch+2}")
            configs['start_swa'] = last_epoch + 2

        swa = SWA(start_epoch=configs['start_swa'], 
              lr_schedule='manual', # other options are constant and cyclic (cycle between lr1 and lr2)
              swa_lr=configs['swa_lr'], 
              swa_lr2=configs['swa_lr2'],
              swa_freq=5,
              verbose=1)
        callbacks.append(swa)
#    backupandrestore = tf.keras.callbacks.BackupAndRestore(backup_dir=model_outdir+"/tmp_backup", delete_checkpoint=False)

    assert len(configs['activations']) == len(configs['layer_sizes']),'the number of activations must be same as the number of layer'
    if configs['activations'][-1] != 'linear':
        print(f"You have set the last layer to {configs['activations'][-1]} but must be set to linear")
        print(f'we set it to linear')
        configs['activations'][-1] = 'linear'
    

    
    #callback_earlystop = tf.keras.callbacks.EarlyStopping(monitor='RMSE_F',
    #                                          patience=decay_step,
    #                                          baseline = 1e-5
   #                                             )
    
#    train_writer = tf.summary.create_file_writer(model_outdir+'/train')
    
    #checkpoint_path = model_outdir+"/models/ckpts-{epoch:04d}-{val_loss:.5f}.keras"
#    checkpoint_path = model_outdir+"/models/ckpts-{epoch:04d}.keras"
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:  # Use the Default Strategy
        strategy = tf.distribute.get_strategy()

    global_batch_size = (configs['batch_size'] *
                     strategy.num_replicas_in_sync)
    # Data preparation
    print('Preparing data...')
    if configs['include_vdw']:
        rc = np.max([configs['rc_rad'], configs['rmax_d']])
    else:
        rc = configs['rc_rad']
    
    _pbc = configs['pbc'] 
    pbc = [True,True,True] if _pbc else [False,False,False]

    atomic_energy = configs['atomic_energy']
    if len(atomic_energy)==0:
        try:
            with open (os.path.join(model_outdir,'atomic_energy.json')) as df:
                atomic_energy = json.load(df)
            atomic_energy = np.array([atomic_energy[key] for key in species])
        except:
            atomic_energy = []


    train_data, test_data, species_identity = data_preparation(
        data_dir=configs['data_dir'],
        species=configs['species'], 
        data_format=configs['data_format'],
        energy_key=configs['energy_key'],
        force_key=configs['force_key'],
        rc=rc,
        pbc=configs['pbc'],
        batch_size=global_batch_size,
        test_fraction=configs['test_fraction'],
        atomic_energy=atomic_energy,
        atomic_energy_file=os.path.join(model_outdir,'atomic_energy.json'),
        model_version=configs['model_version'],
        model_dir=model_outdir
        )

    configs['species_identity'] = species_identity
    if os.path.exists(model_outdir+"/tmp_backup"):
        print("Restoring from checkpoint {model_outdir+'/tmp_backup'}")
    backupandrestore = tf.keras.callbacks.BackupAndRestore(backup_dir=model_outdir+"/tmp_backup",
                                                          delete_checkpoint=False)
    #    callbacks.append(backupandrestore)

    with strategy.scope():
        #model = make_or_restore_model(model_outdir,configs,optimizer)
        model = get_compiled_model(configs,optimizer)
        #model = mBP_model(configs)
        #model.compile(optimizer=optimizer, 
        #          loss="mse", 
        #          metrics=["MAE", 'loss'])

#    model.save_weights(checkpoint_path.format(epoch=0))
    try:
        model.fit(train_data,
             epochs=configs['num_epochs'],
             batch_size=global_batch_size,
             validation_data=test_data,
             validation_freq=10,
             callbacks=[callbacks, backupandrestore])
    except:
        pass

    model.fit(train_data,
             epochs=configs['num_epochs'],
             batch_size=global_batch_size,
             validation_data=test_data,
             validation_freq=10,
             callbacks=callbacks)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='create ML model')
    parser.add_argument('-c', '--config', type=str,
                        help='configuration file', required=True)
    args = parser.parse_args()


    import yaml
    with open(args.config) as f:
        configs = yaml.safe_load(f)

    print(configs)

    create_model(configs)
