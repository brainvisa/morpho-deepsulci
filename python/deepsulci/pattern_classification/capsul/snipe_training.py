'''
SNIPE Training Module
'''

from __future__ import print_function
from ...deeptools.dataset import extract_data
from ..method.snipe import SnipePatternClassification
from capsul.api import Process
from soma import aims

import traits.api as traits
import numpy as np
import json


class PatternSnipeTraining(Process):
    '''
    Process to train the Scoring by Non-local Image PAtch Estimator (SNIPE)
    based model (Coupé et al., 2012) to recognize a searched fold pattern.

    This process consists of two steps. The second test depends on the first
    step. However, they can be started independently if the previous steps have
    already been completed.

    The first step is to extract from the graphs the data useful for training
    the SNIPE-based model (buckets and labels).
    These data are stored in Jason files (traindata_file).

    The second step allows to set the hyperparameters (number of time the
    Optimized Patch Match (OPM) agorithm is run and patch sizes used) by 3-fold
    cross-validation.
    These hyperparameters are saved in the Jason file param_file.

    **Warning:** The searched pattern must have been manually labeled on the
    graphs of the training database containing it.

    '''
    def __init__(self):
        super(PatternSnipeTraining, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(
            output=False, desc='training base graphs')))
        self.add_trait('pattern', traits.Str(
            output=False, desc='vertex name representing the'
                               ' searched pattern'))
        self.add_trait('names_filter', traits.ListStr(
            output=False, desc='list of vertex names defining the region of'
                               ' interest'))
        self.add_trait('num_cpu', traits.Int(
            1, output=False, optional=True, default=1,
            desc='number of processes that can be used to parallel the'
                 ' calculations'))
        self.add_trait('step_1', traits.Bool(
            True, output=False, optional=True,
            desc='perform the data extraction step from the graphs'))
        self.add_trait('step_2', traits.Bool(
            True, output=False, optional=True,
            desc='perform the hyperparameter tuning step'
                 ' (OPM number, patch sizes)'))

        self.add_trait('traindata_file', traits.File(
            output=True, desc='file (.json) storing the data extracted'
                              ' from the training base graphs'))
        self.add_trait('param_file', traits.File(
            output=True, desc='file (.json) storing the hyperparameters'
                              ' (OPM number, patch sizes)'))

    def _run_process(self):

        # Compute bounding box and labels
        if self.step_1:
            print()
            print('--------------------------------')
            print('---         STEP (1/2)       ---')
            print('--- EXTRACT DATA FROM GRAPHS ---')
            print('--------------------------------')
            print()

            dict_label, dict_bck, dict_bck_filtered = {}, {}, {}
            for gfile in self.graphs:
                print(gfile)
                graph = aims.read(gfile)
                side = gfile[gfile.rfind('/')+1:gfile.rfind('/')+2]
                data = extract_data(graph, flip=True if side == 'R' else False)
                label = 0
                fn = []
                for name in data['names']:
                    if name.startswith(self.pattern):
                        label = 1
                    fn.append(sum([1 for n in self.names_filter if name.startswith(n)]))
                bck_filtered = np.asarray(data['bck2'])[np.asarray(fn) == 1]
                dict_label[gfile] = label
                dict_bck[gfile] = data['bck2']
                dict_bck_filtered[gfile] = [list(p) for p in bck_filtered]

            # save in parameters file
            param = {'dict_bck': dict_bck,
                     'dict_bck_filtered': dict_bck_filtered,
                     'dict_label': dict_label}
            with open(self.traindata_file, 'w') as f:
                json.dump(param, f)
        else:
            with open(self.traindata_file) as f:
                param = json.load(f)
            dict_bck = param['dict_bck']
            dict_bck_filtered = param['dict_bck_filtered']
            dict_label = param['dict_label']

        method = SnipePatternClassification(
            pattern=self.pattern, names_filter=self.names_filter,
            dict_bck=dict_bck, dict_bck_filtered=dict_bck_filtered,
            dict_label=dict_label, num_cpu=self.num_cpu)

        # Inner cross validation - fix learning rate / momentum
        if self.step_2:
            print()
            print('---------------------------')
            print('---      STEP (2/2)     ---')
            print('--- FIX HYPERPARAMETERS ---')
            print('---------------------------')
            print()
            method.find_hyperparameters(self.graphs, self.param_file)
