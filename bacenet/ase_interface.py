import os
'''
The following command is used to supress the following wied message
I tensorflow/core/framework/local_rendezvous.cc:421] Local rendezvous recv item cancelled. Key hash:
'''
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np

from ase.calculators.calculator import FileIOCalculator,Calculator, all_changes
from ase.io import write
from data.unpack_tfr_data import unpack_data
import tensorflow as tf
from models.model import BACENET
from models.model_lchannels import BACENET as BACENET_lc

from data.data_processing import data_preparation, atomic_number

import sys, yaml,argparse, json
import numpy as np
import bacenet.train as train
from pathlib import Path



import os

class bacenet_Calculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress','charges', 'zstar']
    #implemented_properties = ['energy', 'forces']
#    ignored_changes = {'pbc'}
#    discard_results_on_any_change = True

#    fileio_rules = FileIOCalculator.ruleset(
#        stdout_name='weighted_mBP.out')

    def __init__(self, 
                 config=None,efield=None,
                 **kwargs):
        """Construct weighted Behler Parrinello  calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: this is currently empty
        native keywords.
        """

        super().__init__(**kwargs)
        
        #Read in model parameters
        #I am parsing yaml files with all the parameters

        #
        with open(config) as f:
            configs = yaml.safe_load(f)

        _configs = train.default_config()
        for key in _configs:
            if key not in configs.keys():
                configs[key] = _configs[key]
        species_chi0, species_J0 = train.estimate_species_chi0_J0(configs['species'])
        if configs['scale_J0'] is None:
            configs['scale_J0'] = tf.ones_like(species_chi0)
        configs['species_chi0'] = species_chi0
        configs['species_J0'] = species_J0
        
        if efield is not None:
            print('applying electric field of: ',efield)
            self.efield = efield
            configs['efield'] = tf.cast(efield, tf.float32)

        if configs['include_vdw']:
            rc = np.max([configs['rc_rad'],configs['rmax_d']])
        else:
            rc = configs['rc_rad']

        _model_call = BACENET
        if configs['model_version'] != 'linear':
            _model_call = BACENET_lc

        atomic_energy = configs['atomic_energy']
        if len(atomic_energy)==0:
            with open (os.path.join(configs['model_outdir'],'atomic_energy.json')) as df:
                self.atomic_energy_dic = json.load(df)

        #print(atomic_energy)
        self.atomic_energy = np.array([self.atomic_energy_dic[key] for key in configs['species']])
        self.species_identity = np.array([atomic_number(key) for key in configs['species']])
        configs['batch_size'] = 1
        model_outdir = configs['model_outdir']
        ckpts = [os.path.join(configs['model_outdir']+"/models", x.split('.index')[0]) 
                 for x in os.listdir(configs['model_outdir']+"/models") if x.endswith('index')]
        ckpts.sort()
    #    ckpts = [os.path.join(model_outdir+"/models", x.split('.weights.h5')[0])
    #         for x in os.listdir(model_outdir+"/models") if x.endswith('h5')]
        #ckpts.sort()
        ckpts_idx = [int(ck.split('-')[-1].split('.')[0]) for ck in ckpts]
        #ckpts_idx = [int(ck.split('-')[-1]) for ck in ckpts]
        ckpts_idx.sort()
        epoch = ckpts_idx[-1]
        idx=f"{epoch:04d}"
        self.ckpt = model_outdir+"/models/"+f"ckpts-{idx}.ckpt"
        #print(f'##################################################################')
        #print(f'calculation are performed with the model:')
        #print(f'{self.ckpt}')
        #print(f'##################################################################')


        #self.ckpt = model_outdir+"/models/"+f"ckpts-{idx}.weights.h5"
        species_identity = [atomic_number(s) for s in configs['species']]
        #print(species_identity)
                
        configs['species_identity'] = species_identity
#        print(species_identity, len(species_identity))
        self.model_call = _model_call(configs)
        self.model_call.load_weights(self.ckpt).expect_partial()
        #weights = self.model.get_weights()
        #print(weights[0])

        #self.model.load_weights(ckpts[-1]).expect_partial()
        self.configs = configs
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms)
        configs = self.configs
        data, test_data, \
                species_identity, _ = \
                data_preparation([atoms], 
                                 configs['species'],
                                 'ase',
                                 configs['energy_key'], 
                                 configs['force_key'],
                                 configs['rc_rad'],
                                 [True]*3, 
                                 1,
                                 test_fraction=0,
                                 atomic_energy=self.atomic_energy,
                                 atomic_energy_file=os.path.join(configs['model_outdir'],'atomic_energy.json'),
                                 model_version=configs['model_version'],
                                 model_dir='./',
                                 evaluate_test=-1,
                                 max_workers=1)
        #print(self.ckpt)
        
        #inputs = unpack_data(list(data)[0])
        inputs = list(data)[0]
        outs = self.model_call(inputs)
        e0 = np.sum([self.atomic_energy_dic[s] for s in atoms.get_chemical_symbols()])
        energy = outs['energy'].numpy()[0] + e0
        forces = tf.squeeze(outs['forces']).numpy()
        stress = tf.squeeze(outs['stress']).numpy()
        charges = tf.squeeze(outs['charges']).numpy()
        #zstar = tf.squeeze(outs['Zstar']).numpy()
        #atoms.set_array('zstar',zstar)

        '''
        configs['species_identity'] = species_identity
        if configs['coulumb']:
            e_ref, e_pred, metrics, force_ref, force_pred,nat,charges,stress = self.model.predict(data)
        else:
            e_ref, e_pred, metrics, force_ref, force_pred,nat,stress = self.model.predict(data)
            charges = np.zeros_like(atoms.get_chemical_symbols())
        '''

#        print(e_pred, force_pred)
        #there is the batch axis
        self.results = {
            "energy":energy,
            "forces":forces,
            "stress":stress,
            "charges":charges}
