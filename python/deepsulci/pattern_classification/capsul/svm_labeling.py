from __future__ import print_function
from ..method.svm import SVMPatternClassification
from capsul.api import Process
import traits.api as traits
import pandas as pd
import time
import json


class PatternSVMLabeling(Process):
    def __init__(self):
        super(PatternSVMLabeling, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False)))
        self.add_trait('train_graphs', traits.List(traits.File(output=False)))
        self.add_trait('param_file', traits.File(output=False))
        self.add_trait('result_file', traits.File(output=True))

    def _run_process(self):
        start_time = time.time()
        with open(self.param_file) as f:
            param = json.load(f)
        method = SVMPatternClassification(
            pattern=param['pattern'], names_filter=param['names_filter'],
            C=param['C'], gamma=param['gamma'], trans=param['trans'])
        method.learning(self.train_graphs)
        y_pred = method.labeling(self.graphs)
        result = pd.DataFrame(index=[str(g) for g in self.graphs])
        result['y_pred'] = y_pred
        result.to_csv(self.result_file)
        print('Labeling took %i sec.' % (time.time() - start_time))
