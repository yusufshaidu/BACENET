#from __future__ import absolute_import, division, print_function, unicode_literals
import os
'''
The following command is used to supress the following wied message
I tensorflow/core/framework/local_rendezvous.cc:421] Local rendezvous recv item cancelled. Key hash:
'''
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf
from swa.tfkeras import SWA
import math 
import os, json
from tensorflow.keras.optimizers import Adam
from data.unpack_tfr_data import unpack_data
from networks.networks import Networks
import functions.helping_functions as help_fn

from mendeleev import element
import argparse
from data.data_processing import data_preparation
from models import model
from models import model_lchannels
from models import model_run
import logging
import yaml

#tf.config.run_functions_eagerly(True)
def default_config():
    return {
        'layer_sizes': None,
        'nelement': 118,
        'save_freq': 'epoch',
        'zeta': 4,
        'rc_rad': 6,
        'Nrad': 8,
        'n_bessels': None,
        'fcost': 1.0,
        'fcost_swa': 1.0,
        'ecost': 1.0,
        'qcost': 1.0,
        'ecost_swa': 1.0,
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
        'clip_value': 100,
        'species_layer_sizes': [],
        'species_correlation': 'dot',
        'radial_layer_sizes': [128, 128],
        'learn_radial': True,
        'activations': None,
        'coulumb': False,
        'accuracy': 1e-6,
        'total_charge': 0.0,
        'features': False,
        'normalize': False,
        'efield': None,
        'oxidation_states': None,
        'is_training': True,
        'n_lambda': 2,
        'lambda_act': 'tanh',
        'self_correction': False,
        'start_swa_global_step': 1000000000,
        'scale_J0': None,
        'scale_chi0': None,
        'per_atom': False,
        'pqeq': False,
        'species_nelectrons': None,
        'debug': False,
        'initial_global_step': 0,
        'gaussian_width_scale': 1.0,
        'max_width': 3.0,
        'linearize_d': 0,
        'nshells': 2,
        'anisotropy': False,
        'initial_shell_spring_constant': False,
        'sawtooth_PE': False,
        'central_atom_id': 1, #This should be set
        'linear_d_terms': False,
        'd0': 0.001,
        'P_in_cell': False,
        'learn_species_nelectrons': False,
        'learnable_gaussian_width': False
            }

def estimate_species_chi0_J0(species):
    IE = tf.constant([element(sym).ionenergies[1]
        for sym in species], dtype=tf.float32)
    EA = tf.constant([0.0 if not element(sym).electron_affinity else
        element(sym).electron_affinity for sym in species], dtype=tf.float32)

    species_hardness = IE - EA
    species_electronegativity = 0.5 * (IE + EA)
    return (species_electronegativity, species_hardness)

def get_compiled_model(configs,optimizer,example_input):
    model_version = configs['model_version']
    if model_version != 'linear':
        #_model = model_lchannels.BACENET(configs=configs)
        _model = model_run.BACENET(configs=configs)
    else:
        _model = model.BACENET(configs=configs)

    #We should do something for tensorflow > 2.15.0
    #fake call to build the model
    # This will “dry run” through call() and allocate weights:
    #example_input = unpack_data(example_input)
    outs = _model(example_input, training=False)
    _model.compile(optimizer=optimizer,
                  loss=help_fn.quad_loss,
                  #loss=help_fn.mae_loss,
                  #loss_f = help_fn.force_loss_mae)
                  loss_f = help_fn.force_loss,
                  loss_q = help_fn.charge_loss)
    print(_model.summary())
    return _model


#This is an example from tensorflow not currently used
#TODO
def make_or_restore_model(model_outdir,configs,optimizer,
                          example_input):
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    model_version = configs['model_version']
    checkpoint_dir = model_outdir + "/models/"
    checkpoints = [checkpoint_dir + name for name in os.listdir(checkpoint_dir) if name.endswith('index')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        latest_checkpoint = latest_checkpoint.split('.index')[0]
        last_ckpt_idx = int(latest_checkpoint.split('-')[-1].split('.')[0])
        print("Restoring from", latest_checkpoint)
        if model_version != 'linear':
            #_model = model_lchannels.BACENET(configs=configs)
            _model = model_run.BACENET(configs=configs)
        else:
            _model = model.BACENET(configs=configs)

        outs = _model(example_input, training=False)
        #checkpoint = tf.train.Checkpoint(model=_model, optimizer=optimizer)
        #latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
        #if latest_ckpt:
        #    checkpoint.restore(latest_ckpt).expect_partial()

        #load and set weights manually
        _model.load_weights(latest_checkpoint).expect_partial()
        # probably do this in the most naive way
        weights = _model.get_weights()
        _model.set_weights(weights)
        _model.compile(optimizer=optimizer,
               loss=help_fn.quad_loss,
               loss_f=help_fn.force_loss,
               loss_q = help_fn.charge_loss)

        print(_model.summary())
        return _model, last_ckpt_idx
    else:
        last_ckpt_idx = 0
    print("Creating a new model")
    
    return get_compiled_model(configs,optimizer,example_input), last_ckpt_idx


def opt(configs, lr_schedule, init_step=0):
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
             weight_decay=0.001, 
             clipnorm=None,
             clipvalue=configs['clip_value'],
             amsgrad=False,
             name='adam')

    #optimizer.lr = optimizer.learning_rate

    optimizer.iterations.assign(init_step)
    return optimizer
class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        #self.model._training_state = None
#        self.model._training_state["epoch"] = epoch
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))


def create_model(configs):
    '''
    # Set memory growth before any GPU work is done
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print("Failed to set memory growth:", e)
    else:
        tf.config.set_visible_devices([], 'GPU')
        tf.config.threading.set_inter_op_parallelism_threads(16)
        tf.config.threading.set_intra_op_parallelism_threads(16)

    '''
    #Read in model parameters
    #I am parsing yaml files with all the parameters
    #
    _configs = default_config()
    
    for key in _configs:
        if key not in configs.keys():
            configs[key] = _configs[key]
    # save training configuration to a YAML file
    model_outdir = os.path.abspath(configs['outdir'])
    configs['outdir'] = model_outdir
    if not os.path.exists(model_outdir):
        os.makedirs(model_outdir, exist_ok=True)
    with open(f'{model_outdir}/train_config.yaml', 'w') as file:
        yaml.dump(configs, file, sort_keys=False)

    species_chi0, species_J0 = estimate_species_chi0_J0(configs['species'])
    if configs['scale_J0'] is None:
        configs['scale_J0'] = tf.ones_like(species_chi0)
    if configs['scale_chi0'] is None:
        configs['scale_chi0'] = tf.ones_like(species_chi0)

    configs['species_chi0'] = species_chi0
    configs['species_J0'] = species_J0 #* configs['scale_J0']
    print(f"the values of chi0 used are {configs['scale_chi0']*species_chi0}")
    print(f"the values of J0 used are {configs['scale_J0']*species_J0}")


    lr_schedule = configs['initial_lr']
    # Learning rate schedule
    if not configs['fixed_lr']:
        '''
        cb_ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='RMSE_F',# energy rmse 
        factor=configs['decay_rate'],
        patience=configs['decay_step'],
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=configs['min_lr'],
        )

        '''
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=configs['initial_lr'],
        decay_steps=configs['decay_step'],
        decay_rate=configs['decay_rate'],
        staircase=True,
        name='ExponentialDecay')

    else:
        lr_schedule = configs['initial_lr']
    
    #save checkpoints
    checkpoint_dir = model_outdir+"/models"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir,exist_ok=True)
   
    checkpoint_path = checkpoint_dir+"/ckpts-{epoch:04d}.ckpt"
#    checkpoint_path = checkpoint_dir+"/ckpts-{epoch:04d}.weights.h5"
    #checkpoint_path = checkpoint_dir+"/ckpts-{epoch:04d}.keras"

    #checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                    save_freq=configs['save_freq']),
                   tf.keras.callbacks.TensorBoard(model_outdir, histogram_freq=1,
                                                  update_freq=configs['save_freq']),
                   tf.keras.callbacks.CSVLogger(model_outdir+"/metrics.dat", separator=" ", append=True)]

    #this callback seems to slowdown training by a lot
    #backupandrestore = tf.keras.callbacks.BackupAndRestore(backup_dir=model_outdir+"/tmp_backup",
    #                                                      delete_checkpoint=False, save_freq='epoch')

                 #CustomCallback()]
    #if not configs['fixed_lr']:
    #    callbacks.append(cb_ReduceLROnPlateau)
    


    if os.path.exists(model_outdir+"/tmp_backup"):
        print(f"Restoring from checkpoint {model_outdir}+'/tmp_backup'")
        epochs = np.loadtxt(model_outdir+"/metrics.dat",skiprows=1, usecols=0).tolist()
        if type(epochs) is list: 
            last_epoch = int(epochs[-1])
        else:
            last_epoch = int(epochs)
    else:
        last_epoch = 0
    #configs['start_swa_global_step'] = 1000000000 # if not swa, no need to increase or descrease cutoff for now
    start_swa_step = 1000000000
    if configs['start_swa'] > -1:
        #try:
        #    epochs = np.loadtxt(model_outdir+"/metrics.dat",skiprows=1, usecols=0)
        #    last_epoch = int(epochs[-1])
        #except:
        #    last_epoch = 0
        if configs['start_swa'] < last_epoch:
            print(f"start_epoch_swa must be larger than the last saved epoch but are {configs['start_swa']} and {last_epoch}")
            print(f"setting start_swa to {last_epoch+2}")
            configs['start_swa'] = last_epoch + 2
        #if configs['fcost_swa'] is None:
        #    configs['fcost_swa'] = configs['fcost']
        #if configs['ecost_swa'] is None:
        #    configs['ecost_swa'] = configs['ecost']
        start_swa_step = configs['start_swa'] # we multiply by step_per_epoch later

        swa = SWA(start_epoch=configs['start_swa'], 
              lr_schedule='manual', # other options are constant and cyclic (cycle between lr1 and lr2)
              swa_lr=configs['swa_lr'], 
              swa_lr2=configs['swa_lr2'],
              swa_freq=5,
              verbose=1)
#        callbacks.append(swa)

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
    '''
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
        print(f'found {strategy.num_replicas_in_sync} GPUs!')
    else:  # Use the Default Strategy
        strategy = tf.distribute.get_strategy()
    
    #global_batch_size = (configs['batch_size'] *
    #                 strategy.num_replicas_in_sync)
    '''
    global_batch_size = configs['batch_size']
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

    
    train_data, test_data, species_identity, nsamples = data_preparation(
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
        model_dir=model_outdir,n_epochs=configs['num_epochs'],
        #oxidation_states={configs['species'][i]:configs['oxidation_states'][i] 
        #                  for i in range(len(configs['species']))},
        #nuclei_charges={configs['species'][i]:configs['species_nelectrons'][i] 
        #                  for i in range(len(configs['species']))}
        )

    #for batch in train_data.take(1):
     #   print("One batch:", batch)

    if configs['start_swa'] > -1:
        configs['start_swa_global_step'] = start_swa_step * (nsamples // global_batch_size)
    else:
        configs['start_swa_global_step'] = start_swa_step

    configs['species_identity'] = species_identity

    #with strategy.scope():
    #model = get_compiled_model(configs,
    #                               opt(configs, lr_schedule),list(train_data)[0])
    #create or restore model
    checkpoint_dir = model_outdir + "/models/"

    checkpoints = [checkpoint_dir + name for name in os.listdir(checkpoint_dir) if name.endswith('index')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        latest_checkpoint = latest_checkpoint.split('.index')[0]
        last_ckpt_idx = int(latest_checkpoint.split('-')[-1].split('.')[0])
        init_step = (last_ckpt_idx) * (nsamples // global_batch_size)
    else:
        init_step = 0
    
    configs['initial_global_step'] = init_step
    model, initial_epoch = make_or_restore_model(model_outdir,configs,opt(configs, lr_schedule, init_step=init_step),
                          list(train_data)[0])
    if initial_epoch == 0:
        model.save_weights(checkpoint_path.format(epoch=0))
    # save training configuration to a YAML file
    #with open(f'{model_outdir}/train_config.yaml', 'w') as file:
   #     yaml.dump(configs, file, sort_keys=False)

    #'''
   # model.save(checkpoint_path.format(epoch=0))
    
    try:
        model.fit(train_data,
             epochs=configs['num_epochs'],
             batch_size=global_batch_size,
             #steps_per_epoch = nsamples // global_batch_size, 
             initial_epoch = initial_epoch,
             validation_data=test_data,
             validation_freq=10,
             callbacks=callbacks)
             #callbacks=[callbacks, backupandrestore])
    except:
        pass

    model.fit(train_data,
             epochs=configs['num_epochs'],
             batch_size=global_batch_size,
             #steps_per_epoch=nsamples // global_batch_size, 
             initial_epoch = initial_epoch,
             validation_data=test_data,
             validation_freq=10,
             callbacks=callbacks)

#if __name__ == '__main__':
def main():
    parser = argparse.ArgumentParser(description='create ML model')
    parser.add_argument('-c', '--config', type=str,
                        help='configuration file', required=True)
    args = parser.parse_args()


    import yaml
    with open(args.config) as f:
        configs = yaml.safe_load(f)

    print(configs)

    create_model(configs)
