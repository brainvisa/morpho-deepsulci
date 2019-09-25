'''
SNIPE Labeling Module
'''

from __future__ import print_function
from ..method.snipe import SnipePatternClassification
from capsul.api import Process
import traits.api as traits
import pandas as pd
import time
import json


class PatternSnipeLabeling(Process):
    '''
    Process to recognize a searched fold pattern using the Scoring by Non-local
    Image PAtch Estimator (SNIPE) based model (Coupé et al., 2012).
    '''

    def __init__(self):
        super(PatternSnipeLabeling, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(
            output=False, desc='graphs to classify')))
        self.add_trait('traindata_file', traits.File(
            output=False, desc='file (.json) storing the data extracted'
                               ' from the training base graphs'))
        self.add_trait('param_file', traits.File(
            output=False, desc='file (.json) storing the hyperparameters'
                               ' (OPM number, patch sizes)'))
        self.add_trait('num_cpu', traits.Int(
            1, output=False, optional=True,
            desc='number of processes that can be used to parallel the'
                 ' calculations'))
        self.add_trait('result_file', traits.File(
            output=True, desc='file (.csv) with predicted class (y_pred)'
                              ' for each of the input graphs'))

    def _run_process(self):
        start_time = time.time()
        with open(self.traindata_file) as f:
            traindata = json.load(f)
        with open(self.param_file) as f:
            param = json.load(f)
        method = SnipePatternClassification(
            pattern=param['pattern'], names_filter=param['names_filter'],
            n_opal=param['n_opal'], patch_sizes=param['patch_sizes'],
            dict_bck=traindata['dict_bck'],
            dict_bck_filtered=traindata['dict_bck_filtered'],
            dict_label=traindata['dict_label'],
            num_cpu=self.num_cpu)
        method.learning(param['gfile_list'])
        y_pred, y_score = method.labeling(self.graphs)
        result = pd.DataFrame(index=[str(g) for g in self.graphs])
        result['y_pred'] = y_pred
        result['snipe'] = y_score
        result.to_csv(self.result_file)
        print('Labeling took %i sec.' % (time.time() - start_time))
