"""
SNIPE Training Module
"""

from ...deeptools.dataset import extract_data
from ..method.snipe import SnipePatternClassification
from capsul.api import Process
from soma import aims
from datetime import timedelta

from soma.controller import File, field
import numpy as np
import json
import time


class PatternSnipeTraining(Process):
    """
    Process to train the Scoring by Non-local Image PAtch Estimator (SNIPE)
    based model (Coupe et al., 2012) to recognize a searched fold pattern.

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

    """

    graphs: field(type_=list[File], doc="training base graphs")
    pattern: field(type_=str, doc="vertex name representing the searched pattern")
    names_filter: field(
        type_=list[str],
        doc="list of vertex names defining the region of interest",
    )
    num_cpu: field(
        type_=int,
        doc="number of processes that can be used to parallel the calculations",
    ) = 1
    step_1: field(
        type_=bool,
        optional=True,
        doc="perform the data extraction step from the graphs",
    ) = True
    step_2: field(
        type_=bool,
        optional=True,
        doc="perform the hyperparameter tuning step (OPM number, patch sizes)",
    ) = True
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

    def execution(self, context):

        # Compute bounding box and labels
        if self.step_1:
            print()
            print("--------------------------------")
            print("---         STEP (1/2)       ---")
            print("--- EXTRACT DATA FROM GRAPHS ---")
            print("--------------------------------")
            print()
            start = time.time()

            dict_label, dict_bck, dict_bck_filtered = {}, {}, {}
            for gfile in self.graphs:
                print(gfile)
                graph = aims.read(gfile)
                side = gfile[gfile.rfind("/") + 1 : gfile.rfind("/") + 2]
                data = extract_data(graph, flip=True if side == "R" else False)
                label = 0
                fn = []
                for name in data["names"]:
                    if name.startswith(self.pattern):
                        label = 1
                    fn.append(sum([1 for n in self.names_filter if name.startswith(n)]))
                bck_filtered = np.asarray(data["bck2"])[np.asarray(fn) == 1]
                dict_label[gfile] = label
                dict_bck[gfile] = data["bck2"]
                dict_bck_filtered[gfile] = [list(p) for p in bck_filtered]

            # save in parameters file
            param = {
                "dict_bck": dict_bck,
                "dict_bck_filtered": dict_bck_filtered,
                "dict_label": dict_label,
            }
            with open(self.traindata_file, "w") as f:
                json.dump(param, f)
            end = time.time()
            print()
            print("STEP 1 took %s" % str(timedelta(seconds=int(end - start))))
            print()
        else:
            with open(self.traindata_file) as f:
                param = json.load(f)
            dict_bck = param["dict_bck"]
            dict_bck_filtered = param["dict_bck_filtered"]
            dict_label = param["dict_label"]

        method = SnipePatternClassification(
            pattern=self.pattern,
            names_filter=self.names_filter,
            dict_bck=dict_bck,
            dict_bck_filtered=dict_bck_filtered,
            dict_label=dict_label,
            num_cpu=self.num_cpu,
        )

        # Inner cross validation - fix learning rate / momentum
        if self.step_2:
            print()
            print("---------------------------")
            print("---      STEP (2/2)     ---")
            print("--- FIX HYPERPARAMETERS ---")
            print("---------------------------")
            print()
            start = time.time()
            method.find_hyperparameters(self.graphs, self.param_file)
            end = time.time()
            print()
            print("STEP 2 took %s" % str(timedelta(seconds=int(end - start))))
            print()
