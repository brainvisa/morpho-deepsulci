# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:06:07 2019

@author: lb251181
"""

from brainvisa.processes import *
from brainvisa.processing import capsul_process

name = 'Deep labeling'
userLevel = 0

base_class = capsul_process.CapsulProcess
capsul_process = 'deepsulci.sulci_labeling.capsul.labeling'
