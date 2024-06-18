import numpy as np

from ase.calculators.calculator import FileIOCalculator,Calculator, all_changes
from ase.io import write
from .model import mBP_model
from .data_processing import data_preparation, atomic_number
#import tensorflow as tf


import os

#class mBP_Calculator(FileIOCalculator):
class mBP_Calculator(Calculator):
    #implemented_properties = ['energy', 'forces','stress']
    implemented_properties = ['energy', 'forces']
#    ignored_changes = {'pbc'}
#    discard_results_on_any_change = True

#    fileio_rules = FileIOCalculator.ruleset(
#        stdout_name='weighted_mBP.out')

    def __init__(self, 
                 config=None,
                 **kwargs):
        """Construct weighted Behler Parrinello  calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' and 'smearing' or any of ELK'
        native keywords.
        """

        super().__init__(**kwargs)
        
        #Read in model parameters
        #I am parsing yaml files with all the parameters
        #
        import yaml
        with open(config) as f:
            configs = yaml.safe_load(f)

        self.layer_sizes = configs['layer_sizes']
        self.zeta = configs['zeta']
        self.thetaN = configs['thetaN']
        self.RsN_rad = configs['RsN_rad']
        self.RsN_ang = configs['RsN_ang']
        self.rc_rad = configs['rc_rad']
        self.rc_ang = configs['rc_ang']
        self.nelement = configs['nelement']
        #estimate initial parameters
        self.width_ang = self.RsN_ang * self.RsN_ang / (self.rc_ang-0.25)**2
        self.width = self.RsN_rad * self.RsN_rad / (self.rc_rad-0.25)**2
        pbc = configs['pbc']
        self.energy_key = 'energy'
        self.force_key = 'force'
        if pbc:
            self.pbc = [True,True,True]
        else:
            self.pbc = [False,False,False]
        #activations are basically sigmoid and linear for now
        self.activations = ['sigmoid', 'sigmoid', 'linear']
        self.species = configs['species']
        self.batch_size = 1
        self.model_outdir = configs['model_outdir']

        self.data_format = 'ase'
        try:
            self.atomic_energy = configs['atomic_energy']
        except:
            self.atomic_energy = []
        species_identity = [atomic_number(s) for s in self.species]
        self.model = mBP_model(self.layer_sizes,
                          self.rc_rad, species_identity, self.width, 1,
                          self.activations,
                          self.rc_ang,self.RsN_rad,self.RsN_ang,
                          self.thetaN,self.width_ang,self.zeta,
                          pbc=self.pbc,
                          nelement=self.nelement)
        ckpts = [os.path.join(self.model_outdir, x.split('.index')[0]) for x in os.listdir(self.model_outdir) if x.endswith('index')]

        self.model.load_weights(ckpts[0]).expect_partial()

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms)

        data, _, _ = data_preparation([atoms], self.species, self.data_format,
                         self.energy_key, self.force_key,
                         self.rc_rad, self.rc_ang, self.pbc, 1,
                         test_fraction=0,
                         atomic_energy=self.atomic_energy)

#        print(data)
#        print(model(data))
        e_ref, e_pred, metrics, force_ref, force_pred,nat = self.model.predict(data)
#        print(e_pred)
        self.results = {
            "energy": e_pred[0],
            "forces": force_pred[0]}
#            "stress":self.calculate_numerical_stress(atoms)} 
            #"stress":self.calculate_numerical_stress(atoms)} 

