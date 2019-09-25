'''
SVM Labeling Module
'''

from __future__ import print_function
from ..method.svm import SVMPatternClassification
from capsul.api import Process
import traits.api as traits
import pandas as pd
import time
import json


class PatternSVMLabeling(Process):
    '''
    Process to recognize a searched fold pattern using a Support Vector
    Machine (SVM) classifier.
    '''

    def __init__(self):
        super(PatternSVMLabeling, self).__init__()
        self.add_trait('graphs', traits.List(
            traits.File(output=False), desc='graphs to classify'))
        self.add_trait('clf_file', traits.File(
            output=False, desc='file (.sav) storing the trained SVM'
                               ' classifier'))
        self.add_trait('scaler_file', traits.File(
            output=False, desc='file (.sav) storing the scaler'))
        self.add_trait('param_file', traits.File(
            output=False, desc='file (.json) storing the hyperparameters'
                               ' (C, gamma, initial tranlations)'))
        self.add_trait('result_file', traits.File(
            output=True, desc='file (.csv) with predicted class (y_pred)'
                              ' for each of the input graphs'))

    def _run_process(self):
        start_time = time.time()
        with open(self.param_file) as f:
            param = json.load(f)
        method = SVMPatternClassification(
            pattern=param['pattern'], names_filter=param['names_filter'],
            C=param['C'], gamma=param['gamma'], trans=param['trans'])
        method.load(self.clf_file, self.scaler_file)
        y_pred = method.labeling(self.graphs)
        result = pd.DataFrame(index=[str(g) for g in self.graphs])
        result['y_pred'] = y_pred
        result.to_csv(self.result_file)
        print('Labeling took %i sec.' % (time.time() - start_time))
