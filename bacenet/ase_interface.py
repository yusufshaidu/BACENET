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
from models.model_run import BACENET

from data.data_processing import data_preparation, atomic_number, _process_file

import sys, yaml,argparse, json
import numpy as np
import bacenet.train as train
from pathlib import Path
import mendeleev
from mendeleev import element




import os

class bacenet_Calculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress','charges', 'zstar', 'shell_displacements', 'Pi_a']
    #implemented_properties = ['energy', 'forces']
#    ignored_changes = {'pbc'}
#    discard_results_on_any_change = True

#    fileio_rules = FileIOCalculator.ruleset(
#        stdout_name='weighted_mBP.out')

    def __init__(self, 
                 config=None,efield=None,central_atom_id=0,
                 **kwargs):
        """Construct ACE-like descriptors Behler Parrinello  calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: this is currently empty
        native keywords.
        """

        super().__init__(**kwargs)
        
        #Read in model parameters
        #I am parsing yaml files with all the parameters

        #
        self._properties = ['energy', 'forces', 'stress','charges', 'zstar', 'shell_displacements']
        with open(config) as f:
            configs = yaml.safe_load(f)
        
        _configs = train.default_config()
        for key in _configs:
            if key not in configs.keys():
                configs[key] = _configs[key]
        if central_atom_id > 0:
            configs['central_atom_id'] = central_atom_id
            #configs['P_in_cell'] = False

        species_chi0, species_J0 = train.estimate_species_chi0_J0(configs['species'])
        if configs['scale_J0'] is None:
            configs['scale_J0'] = tf.ones_like(species_chi0)
        if configs['scale_chi0'] is None:
            configs['scale_chi0'] = tf.ones_like(species_chi0)

        configs['species_chi0'] = species_chi0
        configs['species_J0'] = species_J0
        
        if efield is not None:
            #print('applying electric field of: ',efield)
            #self.efield = efield
            configs['efield'] = tf.cast(efield, tf.float32)

        if configs['include_vdw']:
            rc = np.max([configs['rc_rad'],configs['rmax_d']])
        else:
            rc = configs['rc_rad']

        _model_call = BACENET

        atomic_energy = configs['atomic_energy']
        if len(atomic_energy)==0:
            #with open (os.path.join(configs['model_outdir'],'atomic_energy.json')) as df:
            try:
                #model_outdir = os.path.abspath(configs['outdir'])
                model_outdir = configs['outdir']
            except:
                model_outdir = os.path.abspath(configs['model_outdir'])
            with open (os.path.join(model_outdir,'atomic_energy.json')) as df:
                self.atomic_energy_dic = json.load(df)

        #print(atomic_energy)
        self.atomic_energy = np.array([self.atomic_energy_dic[key] for key in configs['species']])
        self.species_identity = np.array([atomic_number(key) for key in configs['species']])
        configs['batch_size'] = 1
        #model_outdir = configs['model_outdir']
        ckpts = [os.path.join(model_outdir+"/models", x.split('.index')[0]) 
                 for x in os.listdir(model_outdir+"/models") if x.endswith('index')]
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
        print(f'##################################################################')
        print(f'calculation are performed with the model:')
        print(f'{self.ckpt}')
        print(f'##################################################################')


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
        """
        atoms: Atoms object
            Contains positions, unit-cell, ...
        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces'
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these five: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.
        """
        properties = self._properties
        Calculator.calculate(self, atoms, properties, system_changes)
        configs = self.configs
        # C6 are in Ha * au^6
        to_eV = 27.211324570273 * 0.529177**6
        C6_spec = {ss:element(ss).c6_gb * to_eV for ss in configs['species']}
        covalent_radii = {x:element(x).covalent_radius*0.01 for x in configs['species']}

        #file, data_format, species, atomic_energy, C6_spec, energy_key, force_key, rc, evaluate_test,covalent_radii = args
        args = (atoms, 'ase', configs['species'], 
               self.atomic_energy, C6_spec,
               configs['energy_key'],configs['force_key'],
               configs['rc_rad'], -1, covalent_radii)
        _data = _process_file(args)
        data = {}
        for key in _data.keys():
            #add the batch dimension
            values = tf.convert_to_tensor(_data[key], dtype=tf.float32)
            data[key] = tf.expand_dims(tf.reshape(values, [-1]), axis=0)
#            print(key)

        _outs = self.model_call.predict(data, batch_size=1, verbose=0)
        outs = _outs[-1]
        e0 = np.sum([self.atomic_energy_dic[s] for s in atoms.get_chemical_symbols()])

        #there is the batch axis
        self.results = {
            "energy":outs['energy'][0] + e0,
            "forces":outs['forces'][0],
            "stress":outs['stress'][0],
            "charges":outs['charges'][0],
            "shell_displacements":outs['shell_disp'][0],
            "Pi_a": outs['Pi_a'][0],
            "E1": outs['E1'][0],
            "E2": outs['E2'][0],
            "E_d2": outs['E_d2'][0],
            "Zstar": outs['Zstar'][0],
            "epsilon": outs['epsilon'][0],
            }
