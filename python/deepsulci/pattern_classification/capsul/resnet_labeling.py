"""
ResNet Labeling Module
"""

from ..method.resnet import ResnetPatternClassification
from capsul.api import Process
from soma.controller import File, field
import time
import json


class PatternDeepLabeling(Process):
    """
    Process to recognize a searched fold pattern using a 3D-ResNet-18.
    """

    graphs: field(type_=list[File], doc="graphs to classify")
    model_file: field(type_=File, doc="file (.mdsm) storing neural network parameters")
    param_file: field(
        type_=File,
        doc="file (.json) storing the hyperparameters"
        " (bounding_box, learning rate and momentum)",
    )
    result_file: field(
        type_=File,
        write=True,
        doc="file (.csv) with predicted class (y_pred)" " for each of the input graphs",
    )

    def execution(self, context):
        start_time = time.time()
        with open(self.param_file) as f:
            param = json.load(f)
        method = ResnetPatternClassification(param["bounding_box"])
        method.load(self.model_file)
        result = method.labeling(self.graphs)
        result.to_csv(self.result_file)
        print("Labeling took %i sec." % (time.time() - start_time))
