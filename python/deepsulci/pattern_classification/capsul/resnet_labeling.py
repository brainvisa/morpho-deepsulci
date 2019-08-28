from __future__ import print_function
from ..method.resnet import ResnetPatternClassification
from capsul.api import Process
import traits.api as traits
import time


class PatternDeepLabeling(Process):
    def __init__(self):
        super(PatternDeepLabeling, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False)))
        self.add_trait('model_file', traits.File(output=False))
        self.add_trait('result_file', traits.File(output=True))

    def _run_process(self):
        start_time = time.time()
        method = ResnetPatternClassification()
        method.load(self.model_file)
        result = method.labeling(self.graphs)
        result.to_csv(self.result_file)
        print('Labeling took %i sec.' % (time.time() - start_time))
