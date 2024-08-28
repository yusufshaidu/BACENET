import tensorflow as tf
#from NNP_Directfrom_json_pbc import mBP_model, data_preparation

from data_processing import data_preparation
#from model import mBP_model
#from model_legendre_polynomial import mBP_model
#from model_modified_zchannel import mBP_model
from model_modified_manybody import mBP_model
from model_modified import mBP_model as mBP_model_v1

import os, sys, yaml,argparse,json
import numpy as np
import train

def rescale_params(x, a,b):
    rsmin = np.min(x)
    rsmax = np.max(x)
    #rescale between 0.5 and rc for the angular part
    return a + (b - a) * (x - rsmin) / (rsmax - rsmin + 1e-12)

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
    min_radial_center = configs['min_radial_center']

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
                      body_order=body_order,
                      min_radial_center=min_radial_center)
    
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

    layer_names = ['main','species_encoder', 'radial_width', 'ang_width', 'zeta', 'Rs_rad', 'Rs_ang', 'thetas']
    '''prefix = 'main'
    i=0
    for layer,a in zip(layer_sizes,activations):
        layer_names.append(f'{prefix}_{i}_layer_{layer}_activation_{a}')
        i+=1
    i=0
    prefix = 'species_encoder'
    for layer,a in zip([64,118],['sigmoid', 'sigmoid']):
        layer_names.append(f'{prefix}_{i}_layer_{layer}_activation_{a}')
        i+=1
    prefix = 'radial_width'
    i=0
    for layer,a in zip([RsN_rad],['softplus']):
        layer_names.append(f'{prefix}_{i}_layer_{layer}_activation_{a}')
        i+=1
    prefix = 'ang_width'
    i=0
    for layer,a in zip([RsN_ang],['softplus']):
        layer_names.append(f'{prefix}_{i}_layer_{layer}_activation_{a}')
        i+=1
    prefix = 'zeta'
    i=0
    for layer,a in zip([thetaN],['softplus']):
        layer_names.append(f'{prefix}_{i}_layer_{layer}_activation_{a}')
        i+=1
    prefix = 'Rs_rad'
    i=0
    for layer,a in zip([RsN_rad],['sigmoid']):
        layer_names.append(f'{prefix}_{i}_layer_{layer}_activation_{a}')
        i+=1
    prefix = 'Rs_ang'
    i=0
    for layer,a in zip([RsN_ang],['sigmoid']):
        layer_names.append(f'{prefix}_{i}_layer_{layer}_activation_{a}')
        i+=1
    prefix = 'thetas'
    i=0
    for layer,a in zip([thetaN],['sigmoid']):
        layer_names.append(f'{prefix}_{i}_layer_{layer}_activation_{a}')
        i+=1
    '''
#    print(layer_names, len(layer_names), len(model.layers), model.layers)
    parameters = {}
    for i, layer in enumerate(model.layers):
        if i < 2:
            continue
        w,b = layer.get_weights()
        params = w+b
        if layer_names[i] in ['Rs_rad', 'Rs_ang', 'thetas']:
            params = 1 / (1+np.exp(-params))
            if layer_names[i] == 'Rs_rad':
                a = 0.5
                b = rc_rad
            elif layer_names[i] == 'Rs_ang':
                a = 0.5
                b = rc_ang
            else:
                a = 0
                b = np.pi
            params = rescale_params(params, a,b)
        parameters[layer_names[i]] = params.flatten().tolist()

    out_file = open("parameters.json", "w")

    json.dump(parameters, out_file, indent = 6)
        #print(layer_names[i], w+b)
        #print(layer_names[i], w+b)
#    print(parameters)
 #   weights = model.get_weights()
#    print(len(weights), weights)
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
