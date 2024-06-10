
import tensorflow as tf
from NNP_Directfrom_json_pbc import mBP_model, data_preparation

import os, sys, yaml
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
    outdir = configs['outdir']
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

    model = mBP_model(layer_sizes,
                      rc_rad, species, width, batch_size,
                       params_trainable, activations,
                      rc_ang,RsN_rad,RsN_ang,
                      thetaN,width_ang,zeta,
                      fcost=fcost,
                      pbc=pbc)
    
    #load the last check points
    
    latest = tf.train.latest_checkpoint(model_outdir+"/model")
    
    model.load_weights(latest)
    model.compile()

    
    train_data, test_data = data_preparation(data_dir, species, data_format,
                     energy_key, force_key,
                     rc_rad, rc_ang, pbc, batch_size,
                     test_fraction=test_fraction,
                     atomic_energy=atomic_energy)

    model.predict(train_data)
    e_ref, e_pred, metrics, force_ref, force_pred, nat = model.predict(train_data)
    mae = metrics['MAE']
    rmse = metrics['RMSE']
    print(f'Energy: the training rmse = {rmse} and mae = {mae}')
    mae = metrics['MAE_F']
    rmse = metrics['RMSE_F']
    print(f'Forces: the training rmse = {rmse} and mae = {mae}')

    np.savetxt(os.path.join(outdir, 'energy_last_train.dat'), np.stack([e_ref, e_pred, nat]).T)
    model.predict(test_data)
    e_ref, e_pred, metrics, force_ref, force_pred = model.predict(test_data)
    mae = metrics['MAE']
    rmse = metrics['RMSE']
    print(f'Energy: the test rmse = {rmse} and mae = {mae}')
    mae = metrics['MAE_F']
    rmse = metrics['RMSE_F']
    print(f'Forces: the test rmse = {rmse} and mae = {mae}')

    np.savetxt(os.path.join(outdir, 'energy_last_test.dat'), np.stack([e_ref, e_pred, nat]).T)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='create ML model')
    parser.add_argument('-c', '--config', type=str,
                        help='configuration file', required=True)
    args = parser.parse_args()


    import yaml
    with open(args.config) as f:
        configs = yaml.safe_load(f)

    print(configs)




