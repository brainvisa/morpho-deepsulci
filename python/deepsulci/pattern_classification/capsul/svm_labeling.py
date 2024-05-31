"""
SVM Labeling Module
"""

from ..method.svm import SVMPatternClassification
from capsul.api import Process
from soma.controller import File, field
import pandas as pd
import time
import json


class PatternSVMLabeling(Process):
    """
    Process to recognize a searched fold pattern using a Support Vector
    Machine (SVM) classifier.
    """

    graphs: field(type_=list[File], doc="graphs to classify")
    clf_file: field(
        type_=File,
        write=True,
        doc="file (.sav) storing the trained SVM classifier",
    )
    scaler_file: field(
        type_=File,
        write=True,
        doc="file (.sav) storing the scaler",
    )
    param_file: field(
        type_=File,
        write=True,
        doc="file (.json) storing the hyperparameters"
        " (C, gamma, initial tranlations)",
    )
    result_file: field(
        type_=File,
        write=True,
        doc="ffile (.csv) with predicted class (y_pred)"
        " for each of the input graphs",
    )

    def execution(self, context):
        start_time = time.time()
        with open(self.param_file) as f:
            param = json.load(f)
        method = SVMPatternClassification(
            pattern=param["pattern"],
            names_filter=param["names_filter"],
            C=param["C"],
            gamma=param["gamma"],
            trans=param["trans"],
        )
        method.load(self.clf_file, self.scaler_file)
        y_pred = method.labeling(self.graphs)
        result = pd.DataFrame(index=[str(g) for g in self.graphs])
        result["y_pred"] = y_pred
        result.to_csv(self.result_file)
        print("Labeling took %i sec." % (time.time() - start_time))
