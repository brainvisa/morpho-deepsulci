'''
ResNet Labeling Module
'''

from __future__ import print_function
from ..method.resnet import ResnetPatternClassification
from capsul.api import Process
import traits.api as traits
import time
import json


class PatternDeepLabeling(Process):
    '''
    Process to recognize a searched fold pattern using a 3D-ResNet-18.
    '''

    def __init__(self):
        super(PatternDeepLabeling, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False),
                                             desc='graphs to classify'))
        self.add_trait('model_file', traits.File(
            output=False,
            desc='file (.mdsm) storing neural network parameters'))
        self.add_trait('param_file', traits.File(
            output=False, desc='file (.json) storing the hyperparameters'
                               ' (bounding_box, learning rate and momentum)'))
        self.add_trait('result_file', traits.File(
            output=True, desc='file (.csv) with predicted class (y_pred)'
                              ' for each of the input graphs'))

    def _run_process(self):
        start_time = time.time()
        with open(self.param_file) as f:
            param = json.load(f)
        method = ResnetPatternClassification(param['bounding_box'])
        method.load(self.model_file)
        result = method.labeling(self.graphs)
        result.to_csv(self.result_file)
        print('Labeling took %i sec.' % (time.time() - start_time))
