"""
SNIPE Labeling Module
"""

from ..method.snipe import SnipePatternClassification
from capsul.api import Process
from soma.controller import File, field
import pandas as pd
import time
import json


class PatternSnipeLabeling(Process):
    """
    Process to recognize a searched fold pattern using the Scoring by Non-local
    Image PAtch Estimator (SNIPE) based model (Coupé et al., 2012).
    """

    graphs: field(type_=list[File], doc="graphs to classify")
    traindata_file: field(
        type_=File,
        write=True,
        doc="file (.json) storing the data extracted from the training base graphs",
    )
    param_file: field(
        type_=File,
        write=True,
        doc="file (.json) storing the hyperparameters" " (OPM number, patch sizes)",
    )
    num_cpu: field(
        type_=int,
        doc="number of processes that can be used to parallel the calculations",
    ) = 1
    result_file: field(
        type_=File,
        write=True,
        doc="file (.csv) with predicted class (y_pred) for each of the input graphs",
    )

    def execution(self, context):
        start_time = time.time()
        with open(self.traindata_file) as f:
            traindata = json.load(f)
        with open(self.param_file) as f:
            param = json.load(f)
        method = SnipePatternClassification(
            pattern=param["pattern"],
            names_filter=param["names_filter"],
            n_opal=param["n_opal"],
            patch_sizes=param["patch_sizes"],
            dict_bck=traindata["dict_bck"],
            dict_bck_filtered=traindata["dict_bck_filtered"],
            dict_label=traindata["dict_label"],
            num_cpu=self.num_cpu,
        )
        method.learning(param["gfile_list"])
        y_pred, y_score = method.labeling(self.graphs)
        result = pd.DataFrame(index=[str(g) for g in self.graphs])
        result["y_pred"] = y_pred
        result["snipe"] = y_score
        result.to_csv(self.result_file)
        print("Labeling took %i sec." % (time.time() - start_time))
