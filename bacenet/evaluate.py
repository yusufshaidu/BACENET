import os
'''
The following command is used to supress the following wied message
I tensorflow/core/framework/local_rendezvous.cc:421] Local rendezvous recv item cancelled. Key hash:
'''
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf
from data.data_processing import data_preparation
from models.model_run import BACENET

import sys, yaml,argparse, json
import numpy as np
import bacenet.train as train
from pathlib import Path

def create_model(configs):
    #Read in model parameters
    #I am parsing yaml files with all the parameters
    #
    _configs = train.default_config()

    for key in _configs:
        if key not in configs.keys():
            configs[key] = _configs[key]
    species_chi0, species_J0 = train.estimate_species_chi0_J0(configs['species'])
    if configs['scale_J0'] is None:
        configs['scale_J0'] = tf.zeros_like(species_chi0)
    if configs['scale_chi0'] is None:
        configs['scale_chi0'] = tf.zeros_like(species_chi0)

    configs['species_chi0'] = species_chi0
    configs['species_J0'] = species_J0

    # Data preparation
    print('Preparing data...')
    model_outdir = configs['outdir']
    
    if not os.path.exists(configs['test_outdir']):
        os.mkdir(configs['test_outdir'])
    if configs['include_vdw']:
        rc = np.max([configs['rc_rad'], configs['rmax_d']])
    else:
        rc = configs['rc_rad']

    _pbc = configs['pbc']
    pbc = [True,True,True] if _pbc else [False,False,False]

    with open (os.path.join(model_outdir,'atomic_energy.json')) as df:
        atomic_energy = json.load(df)
        atomic_energy = np.array([atomic_energy[key] for key in configs['species']])

    train_data, test_data, species_identity, _ = data_preparation(
        data_dir=configs['data_dir'],
        species=configs['species'],
        data_format=configs['data_format'],
        energy_key=configs['energy_key'],
        force_key=configs['force_key'],
        rc=rc,
        pbc=configs['pbc'],
        batch_size=configs['batch_size'],
        test_fraction=configs['test_fraction'],
        atomic_energy=atomic_energy,
        atomic_energy_file=os.path.join(model_outdir,'atomic_energy.json'),
        model_version=configs['model_version'],
        model_dir=model_outdir,
        evaluate_test=1
        )

    configs['species_identity'] = species_identity


    #
    epoch = configs['epoch']
    try:
        interval = configs['interval']
    except:
        interval = 1
        print('all saved checkpoints will be evaluated')
    #estimate initial parameters
    #trainable linear model
    fcost = configs['fcost']
    #activations are basically tanh and linear for now
    activations = configs['activations']

    assert len(activations) == len(configs['layer_sizes']),'the number of activations must be same as the number of layer'

    test_fraction = configs['test_fraction']
    if test_fraction < 1.0:
        print(f'you are evaluating only on a {test_fraction*100} % of you test dataset')
    model_call = BACENET
    try:
        error_file = configs['error_file']
    except:
        error_file = 'error_file'

    model = model_call(configs)
    
    #ckpts = [os.path.join(model_outdir+"/models", x.split('.index')[0]) 
    #         for x in os.listdir(model_outdir+"/models") if x.endswith('ckpt')]
    ckpts = [os.path.join(configs['outdir']+"/models", x.split('.index')[0])
                 for x in os.listdir(configs['outdir']+"/models") if x.endswith('index')
             ]
    #ckpts = [os.path.join(model_outdir+"/models", x.split('.weights.h5')[0]) 
    ckpts.sort()

    #print(ckpts)
    
    if epoch == -1: 
        ckpts_idx = [int(ck.split('-')[-1].split('.')[0]) for ck in ckpts]
        #ckpts_idx = [int(ck.split('-')[-1]) for ck in ckpts]
        ckpts_idx.sort()
        epoch = ckpts_idx[-1]
        idx=f"{epoch:04d}"
        ck = [model_outdir+"/models/"+f"ckpts-{idx}.ckpt"]
        #ck = [model_outdir+"/models/"+f"ckpts-{idx}.weights.h5"]
#        model.load_weights(ck).expect_partial()
        ckpts_idx = [epoch]

        #model.load_weights(ckpts[-1]).expect_partial()
        print(f'evaluating {ck}')
    else:
        print('############')
        print('I am evaluating all saved check points: may take some time except some of them are already done!!!')
        print('############')
        ckpts_idx = [int(ck.split('-')[-1].split('.')[0]) for ck in ckpts]
        ckpts_idx.sort()

        ck = [model_outdir+"/models/"+f"ckpts-{idx:04d}.ckpt" for idx in ckpts_idx]


    try:
        errors = np.loadtxt(error_file, skiprows=1).tolist()
    except:
        errors = []

    #print(errors)
    for i,_ck in enumerate(ck):
        

        _epoch = ckpts_idx[i]
        if len(ck) > 1:
            pfile = Path(os.path.join(configs['test_outdir'], f'energy_last_test_{_epoch}.dat'))
            if int(i) % interval != 0 or pfile.is_file():
                continue

        model.load_weights(_ck).expect_partial()
        print(f'evaluating {_epoch} epoch')

        e_ref, e_pred, force_ref, force_pred,nat,_charges,stress, outs = model.predict(test_data)
        shell_disp = outs['shell_disp']

        _f_ref = []
        _f_pred = []
        fmaes = []
        frmses = []
        idx = []
        shell_disp = []
        charges = []
        for i, j in enumerate(nat):
            j = tf.cast(j, tf.int32)
            _f_ref = np.append(_f_ref, force_ref[i][:j])
            _f_pred = np.append(_f_pred, force_pred[i][:j])
            if configs['coulumb']:
                charges = np.append(charges, _charges[i][:j])
                shell_disp = np.append(_shell_disp, _charges[i][:j])
            idx = np.append(idx, np.arange(1,j+1).tolist())
        force_ref = tf.reshape(_f_ref, [-1,3])
        force_pred = tf.reshape(_f_pred, [-1,3])
        diff_f = force_ref - force_pred

        if configs['per_atom']:
            dE = (e_ref - e_pred) / nat
            mae = tf.reduce_mean(tf.abs(dE)).numpy()
            rmse = tf.sqrt(tf.reduce_mean((dE)**2)).numpy()
        else:
            mae = tf.reduce_mean(tf.abs(e_ref-e_pred)).numpy()
            rmse = tf.sqrt(tf.reduce_mean((e_ref-e_pred)**2)).numpy()
        fmae = tf.reduce_mean(tf.abs(diff_f)).numpy()
        frmse = tf.sqrt(tf.reduce_mean(diff_f**2)).numpy()
        errors.append([_epoch,rmse*1000,mae*1000,frmse*1000,fmae*1000])

        print(f'Ermse = {rmse*1000:.3f} and Emae = {mae*1000:.3f} | Frmse = {frmse*1000:.3f} and Fmae = {fmae*1000:.3f}')
        #print(f'Forces: the test rmse = {rmse} and mae = {mae}')
        np.savetxt(os.path.join(configs['test_outdir'], 
                                f'energy_last_test_{_epoch}.dat'), 
                   np.stack([e_ref, e_pred, nat]).T)
        np.savetxt(os.path.join(configs['test_outdir'], 
                                f'forces_last_test_{_epoch}.dat'), 
                   np.stack([idx,force_ref[:,0], force_ref[:,1], 
                            force_ref[:,2],force_pred[:,0], 
                             force_pred[:,1], force_pred[:,2]]).T,
                   fmt="%d %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f")

        if configs['coulumb']:
            np.savetxt(os.path.join(configs['test_outdir'], 
                                    f'charges_{_epoch}.dat'), 
                       np.stack([idx, charges]).T)
            np.savetxt(os.path.join(configs['test_outdir'], 
                                    f'shell_disp_{_epoch}.dat'), 
                       np.stack([idx, shell_disp[:,0],shell_disp[:,1],
                                 shell_disp[:,2]]).T, fmt="%d %10.6f %10.6f %10.6f")
    #print(errors)

    #weights = model.get_weights()
    #print(weights)
    np.savetxt(error_file, np.array(errors), header='E_rmse E_mae F_rmse F_mae', fmt='%10.3f')
    
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
