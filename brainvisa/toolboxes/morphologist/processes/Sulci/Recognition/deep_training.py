# -*- coding: utf-8 -*-
from brainvisa.processes import *
from brainvisa.processing import capsul_process

name = 'Deep training'
userLevel = 0

base_class = capsul_process.CapsulProcess
capsul_process = 'deepsulci.sulci_labeling.capsul.training'

signature = Signature(
    'graphs', ListOf(ReadDiskItem('Cortical folds graph', 'Graph and data')),
    'graphs_notcut',
        ListOf(ReadDiskItem('Cortical folds graph', 'Graph and data')),
)

