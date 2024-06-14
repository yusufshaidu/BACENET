import tensorflow as tf
#from NNP_Directfrom_json_pbc import mBP_model, data_preparation

from data_processing import data_preparation
from model import mBP_model

import os, sys, yaml,argparse
import numpy as np

def create_model(config_file):

    #Read in model parameters
    #I am parsing yaml files with all the parameters
    #
    layer_sizes = configs['layer_sizes']
    zeta = configs['zeta']
    thetaN = configs['thetaN']
    RsN_rad = configs['RsN_rad']
    RsN_ang = configs['RsN_ang']
    rc_rad = configs['rc_rad']
    rc_ang = configs['rc_ang']
    #estimate initial parameters
    width_ang = RsN_ang * RsN_ang / (rc_ang-0.25)**2
    width = RsN_rad * RsN_rad / (rc_rad-0.25)**2
    #trainable linear model
    fcost = configs['fcost']
    #trainable linear model
    params_trainable = configs['params_trainable']

    pbc = configs['pbc']
    if pbc:
        pbc = [True,True,True]
    else:
        pbc = [False,False,False]
    #activations are basically sigmoid and linear for now
    activations = ['sigmoid', 'sigmoid', 'linear']
    species = configs['species']
    batch_size = configs['batch_size']
    model_outdir = configs['model_outdir']
    outdir = configs['test_outdir']
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    data_dir = configs['data_dir']
    data_format = configs['data_format']
    energy_key = configs['energy_key']
    force_key = configs['force_key']
    test_fraction = configs['test_fraction']
    try:
        atomic_energy = configs['atomic_energy']
    except:
        atomic_energy = []
    train_data, test_data, species_identity = data_preparation(data_dir, species, data_format,
                     energy_key, force_key,
                     rc_rad, rc_ang, pbc, batch_size,
                     test_fraction=test_fraction,
                     atomic_energy=atomic_energy)

    model = mBP_model(layer_sizes,
                      rc_rad, species_identity, width, batch_size,
                      activations,
                      rc_ang,RsN_rad,RsN_ang,
                      thetaN,width_ang,zeta,
                      fcost=fcost,
                      params_trainable=True,
                      pbc=pbc)
    
    #load the last check points
    
    ckpts = tf.train.latest_checkpoint(model_outdir+"/models")
#    ckpts = tf.train.load_checkpoint(model_outdir+"/models")
    ckpts = [os.path.join(model_outdir+"/models", x.split('.index')[0]) for x in os.listdir(model_outdir+"/models") if x.endswith('index')]
    ckpts.sort()

    print(ckpts)

    
    model.load_weights(ckpts[-1])
    model.compile()
#    print(model.get_weights())
    e_ref, e_pred, metrics, force_ref, force_pred,nat = model.predict(test_data)
    mae = np.mean(metrics['MAE'])
    rmse = np.mean(metrics['RMSE'])
    print(f'Energy: the test rmse = {rmse} and mae = {mae}')
    mae = np.mean(metrics['MAE_F'])
    rmse = np.mean(metrics['RMSE_F'])
    print(f'Forces: the test rmse = {rmse} and mae = {mae}')

    force_ref = tf.reshape(force_ref, [-1,3])
    force_pred = tf.reshape(force_pred, [-1,3])

    np.savetxt(os.path.join(outdir, 'energy_last_test.dat'), np.stack([e_ref, e_pred, nat]).T)
    np.savetxt(os.path.join(outdir, 'forces_last_test.dat'), np.stack([force_ref[:,0], force_ref[:,1], force_ref[:,2],force_pred[:,0], force_pred[:,1], force_pred[:,2]]).T)

    #model.predict(train_data)
    e_ref, e_pred, metrics, force_ref, force_pred, nat = model.predict(train_data)
    mae = np.mean(metrics['MAE'])
    rmse = np.mean(metrics['RMSE'])
    print(f'Energy: the training rmse = {rmse} and mae = {mae}')
    mae = np.mean(metrics['MAE_F'])
    rmse = np.mean(metrics['RMSE_F'])
    print(f'Forces: the training rmse = {rmse} and mae = {mae}')
    force_ref = tf.reshape(force_ref, [-1,3])
    force_pred = tf.reshape(force_pred, [-1,3])

    np.savetxt(os.path.join(outdir, 'energy_last_train.dat'), np.stack([e_ref, e_pred, nat]).T)
    np.savetxt(os.path.join(outdir, 'forces_last_train.dat'), np.stack([force_ref[:,0], force_ref[:,1], force_ref[:,2],force_pred[:,0], force_pred[:,1], force_pred[:,2]]).T)
    #model.predict(test_data)
    
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
