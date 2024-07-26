import tensorflow as tf
#from NNP_Directfrom_json_pbc import mBP_model, data_preparation

from data_processing import data_preparation
#from model import mBP_model
#from model_legendre_polynomial import mBP_model
#from model_modified_zchannel import mBP_model
from model_modified_manybody import mBP_model
from model_modified import mBP_model as mBP_model_v1

import os, sys, yaml,argparse
import numpy as np
import train

def create_model(configs):

    #Read in model parameters
    #I am parsing yaml files with all the parameters

    _configs = train.default_config()
    for key in _configs:
        if key not in configs.keys():
            configs[key] = _configs[key]

    #
    layer_sizes = configs['layer_sizes']
    zeta = configs['zeta']
    thetaN = configs['thetaN']
    RsN_rad = configs['RsN_rad']
    RsN_ang = configs['RsN_ang']
    rc_rad = configs['rc_rad']
    rc_ang = configs['rc_ang']
    nelement = configs['nelement']
    epoch = configs['epoch']
    #estimate initial parameters
    width_ang = RsN_ang * RsN_ang / (rc_ang-0.25)**2
    width = RsN_rad * RsN_rad / (rc_rad-0.25)**2
    #trainable linear model
    fcost = configs['fcost']
    #trainable linear model
    params_trainable = configs['params_trainable']

    _pbc = configs['pbc']
    print(_pbc)
    if _pbc:
        pbc = [True,True,True]
    else:
        pbc = [False,False,False]
    #activations are basically tanh and linear for now
    activations = ['tanh', 'tanh', 'linear']
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
    atomic_energy = configs['atomic_energy']
    train_zeta = configs['train_zeta']
    include_vdw = configs['include_vdw']
    rmin_u = configs['rmin_u']
    rmax_u = configs['rmax_u']
    rmin_d = configs['rmin_d']
    rmax_d = configs['rmax_d']
    body_order = configs['body_order']
    Nzeta = configs['Nzeta']
    learnable_centers = configs['learnable_centers']
    variable_width = configs['variable_width']

    if include_vdw:
        rc = np.max([rc_rad,rc_ang,rmax_d])
    else:
        rc = np.max([rc_rad,rc_ang])

    model_call = mBP_model
    print('model_version' in list(configs.keys()))
    if 'model_version' in list(configs.keys()):
        model_v = configs['model_version']
        if model_v == 'v1':
            model_call = mBP_model_v1

    nspec_embedding = configs['nspec_embedding']

    train_data, test_data, species_identity = data_preparation(data_dir, species, data_format,
                     energy_key, force_key,
                     rc, pbc, batch_size,
                     test_fraction=test_fraction,
                     atomic_energy=atomic_energy)

    model = model_call(layer_sizes,
                      rc_rad, species_identity, width, batch_size,
                      activations,
                      rc_ang,RsN_rad,RsN_ang,
                      thetaN,width_ang,zeta,
                      fcost=fcost,
                      params_trainable=True,
                      pbc=_pbc,
                      nelement=nelement,
                      train_zeta=train_zeta,
                      nspec_embedding=nspec_embedding,
                      include_vdw=include_vdw,
                      rmin_u=rmin_u,rmax_u=rmax_u,
                      rmin_d=rmin_d,rmax_d=rmax_d,
                      Nzeta=Nzeta,
                      learnable_centers=learnable_centers,
                      variable_width=variable_width,
                      body_order=body_order)
    
    #load the last check points
    
#    ckpts = tf.train.latest_checkpoint(model_outdir+"/models")
#    ckpts = tf.train.load_checkpoint(model_outdir+"/models")
    ckpts = [os.path.join(model_outdir+"/models", x.split('.index')[0]) for x in os.listdir(model_outdir+"/models") if x.endswith('index')]
    ckpts.sort()

    #print(ckpts)

    if epoch == -1: 
        ckpts_idx = [int(ck.split('-')[-1].split('.')[0]) for ck in ckpts]
        ckpts_idx.sort()
        epoch = ckpts_idx[-1]
        idx=f"{epoch:04d}"
        ck = model_outdir+"/models/"+f"ckpts-{idx}.ckpt"
        model.load_weights(ck).expect_partial()

        #model.load_weights(ckpts[-1]).expect_partial()
        print(f'evaluating {ck}')
    else:
        idx=f"{epoch:04d}"
        ck = model_outdir+"/models/"+f"ckpts-{idx}.ckpt"
        model.load_weights(ck).expect_partial()
        print(f'evaluating {ck}')

    weights = model.get_weights()

    #print(weights)

    e_ref, e_pred, metrics, force_ref, force_pred,nat = model.predict(test_data)
    _f_ref = []
    _f_pred = []
    for i, j in enumerate(nat):
        j = tf.cast(j, tf.int32)
        _f_ref = np.append(_f_ref, force_ref[i][:j])
        _f_pred = np.append(_f_pred, force_pred[i][:j])

    #force_ref = [tf.reshape(force_ref[:,:tf.cast(i, tf.int32), :], [-1,3]) for i in nat]
    force_ref = tf.reshape(_f_ref, [-1,3])
    #force_pred = [tf.reshape(_f_pred[:,:tf.cast(i, tf.int32), :], [-1,3]) for i in nat]
    force_pred = tf.reshape(_f_pred, [-1,3])
    mae = tf.reduce_mean(tf.abs(e_ref-e_pred))
    rmse = tf.sqrt(tf.reduce_mean((e_ref-e_pred)**2))
    print(f'Energy: the test rmse = {rmse} and mae = {mae}')
    mae = tf.reduce_mean(tf.abs(_f_ref-_f_pred))
    rmse = tf.sqrt(tf.reduce_mean((_f_ref-_f_pred)**2))
    print(f'Forces: the test rmse = {rmse} and mae = {mae}')
    np.savetxt(os.path.join(outdir, f'energy_last_test_{epoch}.dat'), np.stack([e_ref, e_pred, nat]).T)
    np.savetxt(os.path.join(outdir, f'forces_last_test_{epoch}.dat'), np.stack([force_ref[:,0], force_ref[:,1], force_ref[:,2],force_pred[:,0], force_pred[:,1], force_pred[:,2]]).T)
    
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
