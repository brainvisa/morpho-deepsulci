# -*- coding: utf-8 -*-
from brainvisa.processes import *
from brainvisa.processing import capsul_process

name = 'Sulci Deep CNN training'
userLevel = 0

base_class = capsul_process.CapsulProcess
capsul_process = 'deepsulci.sulci_labeling.capsul.training'

signature = Signature(
    'graphs', ListOf(ReadDiskItem('Labelled Cortical folds graph', 'Graph and data', requiredAttributes={'manually_labelled': 'Yes'})),
    'graphs_notcut',
        ListOf(ReadDiskItem('Cortical folds graph', 'Graph and data')),
)

