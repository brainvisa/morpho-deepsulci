from __future__ import print_function
from ..method.resnet import SnipePatternClassification
from capsul.api import Process
import traits.api as traits
import time
import json


class PatternDeepLabeling(Process):
    def __init__(self):
        super(PatternDeepLabeling, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False)))
        self.add_trait('train_graphs', traits.List(traits.File(output=False)))
        self.add_trait('param_file', traits.File(output=False))
        self.add_trait('result_file', traits.File(output=True))

    def _run_process(self):
        start_time = time.time()
        with open(self.param_file) as f:
            param = json.load(f)
        method = SnipePatternClassification(
            pattern=param['pattern'], names_filter=param['names_filter'],
            n_opal=param['n_opal'], patch_sizes=param['patch_sizes'])
        method.learning(self.train_graphs)
        result = method.labeling(self.graphs)
        result.to_csv(self.result_file)
        print('Labeling took %i sec.' % (time.time() - start_time))
