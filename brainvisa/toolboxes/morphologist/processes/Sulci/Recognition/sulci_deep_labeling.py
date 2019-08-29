# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:06:07 2019

@author: lb251181
"""

from brainvisa.processes import *
from brainvisa.processing import capsul_process

name = 'Sulci recognition with Deep CNN'
userLevel = 0

base_class = capsul_process.CapsulProcess
capsul_process = 'deepsulci.sulci_labeling.capsul.labeling'

def validation():
    try:
        import torch
    except ImportError:
        raise ValidationError('PyTorch (torch module) needs to be installed '
                              'to have CNN sulci recognition working')
    shared_db = [db for db in neuroConfig.dataPath
                  if db.expert_settings.ontology == 'shared']
    models = [os.path.exists(p)
              for p in
                  [os.path.join(db.directory, 'models', 'models_2019',
                                'cnn_models', 'sulci_unet_model_left.mdsm')
                   for db in shared_db]]
    if len(models) == 0:
        raise ValidationError('CNN sulci models are not installed')


signature = Signature(
    'roots', ReadDiskItem('Cortex catchment bassins',
                          "aims readable volume formats"),
    'skeleton', ReadDiskItem('Cortex skeleton',
                             "aims readable volume formats"),
)

