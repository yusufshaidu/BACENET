import numpy as np

from ase.calculators.calculator import FileIOCalculator,Calculator, all_changes
from ase.io import write
#import tensorflow as tf
#import tensorflow as tf
from model import mBP_model
from data_processing import data_preparation, atomic_number

import os, sys, yaml,argparse, json
import numpy as np
import train
from pathlib import Path



import os

class wBP_Calculator(Calculator):
    implemented_properties = ['energy', 'forces']
    #implemented_properties = ['energy', 'forces']
#    ignored_changes = {'pbc'}
#    discard_results_on_any_change = True

#    fileio_rules = FileIOCalculator.ruleset(
#        stdout_name='weighted_mBP.out')

    def __init__(self, 
                 config=None,
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
        
        #
        if configs['include_vdw']:
            rc = np.max([configs['rc_rad'],configs['rmax_d']])
        else:
            rc = configs['rc_rad']

        model_call = mBP_model
        atomic_energy = configs['atomic_energy']
        if len(atomic_energy)==0:
            with open (os.path.join(configs['model_outdir'],'atomic_energy.json')) as df:
                atomic_energy = json.load(df)

        self.atomic_energy = np.array([atomic_energy[key] for key in configs['species']])
        self.species_identity = np.array([atomic_number(key) for key in configs['species']])
        configs['batch_size'] = 1
        self.model = model_call(configs)

        ckpts = [os.path.join(configs['model_outdir']+"/models", x.split('.index')[0]) 
                 for x in os.listdir(configs['model_outdir']+"/models") if x.endswith('index')]
        ckpts.sort()

        #ckpts = [os.path.join(self.model_outdir, x.split('.index')[0]) for x in os.listdir(self.model_outdir) if x.endswith('index')]

        self.model.load_weights(ckpts[-1]).expect_partial()
        self.configs = configs
    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms)
        configs = self.configs
        data, test_data, \
                species_identity = \
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
                                 model_version=configs['model_version'],
                                 model_dir=configs['model_outdir'],
                                 evaluate_test=-1,
                                 max_workers=1)

#        print(data)
#        print(model(data))
        e_ref, e_pred, metrics, force_ref, force_pred, nat = self.model.predict(data)
        #print(e_pred)
        self.results = {
            "energy": e_pred[0],
            "forces": force_pred[0]}
            #"stress":self.calculate_numerical_stress(atoms)} 
            #"stress":self.calculate_numerical_stress(atoms)} 

