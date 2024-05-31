"""
SVM Training Module
"""

from ...deeptools.dataset import extract_data
from ..method.svm import SVMPatternClassification
from capsul.api import Process
from soma import aims
from datetime import timedelta

from soma.controller import File, field
import numpy as np
import json
import time


class PatternSVMTraining(Process):
    """
    Process to train a Support Vector Machine (SVM) classifier to recognize a
    searched fold pattern.

    This process consists of three steps. Each step depends on the previous
    step. However, they can be started independently if the previous steps have
    already been completed.

    The first step is to extract from the graphs the data useful for training
    the SVM-based model (buckets and labels).
    These data are stored in Jason files (traindata_file).

    The second step allows to set the hyperparameters (C, gamma and
    translations applied to the patches before their registration with the
    Iterative Closest Point (ICP) algorithm) by 3-fold cross-validation.
    These hyperparameters are saved in the Jason file param_file.

    The third step is to train the SVM on the data.
    The model is saved in clf_file and the scaler allowing to standardize the
    features in scaler_file.

    **Warning:** The searched pattern must have been manually labeled on the
    graphs of the training database containing it.

    """

    graphs: field(type_=list[File], doc="training base graphs")
    pattern: field(type_=str, doc="vertex name representing the searched pattern")
    names_filter: field(
        type_=list[str],
        doc="list of vertex names used for the registration of the patches",
    )
    step_1: field(
        type_=bool,
        optional=True,
        doc="perform the data extraction step from the graphs",
    ) = True
    step_2: field(
        type_=bool,
        optional=True,
        doc="perform the hyperparameter tuning step (C, gamma, initial translations)",
    ) = True
    step_3: field(
        type_=bool,
        optional=True,
        doc="perform the model training step",
    ) = True
    traindata_file: field(
        type_=File,
        write=True,
        doc="file (.json) storing the data extracted from the training base graphs",
    )
    param_file: field(
        type_=File,
        write=True,
        doc="file (.json) storing the hyperparameters"
        " (C, gamma, initial tranlations)",
    )
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

    def execution(self, context):

        # Compute bounding box and labels
        if self.step_1:
            print()
            print("--------------------------------")
            print("---         STEP (1/3)       ---")
            print("--- EXTRACT DATA FROM GRAPHS ---")
            print("--------------------------------")
            print()
            start = time.time()

            dict_label, dict_bck, dict_bck_filtered = {}, {}, {}
            dict_searched_pattern = {}
            for gfile in self.graphs:
                print(gfile)
                graph = aims.read(gfile)
                side = gfile[gfile.rfind("/") + 1 : gfile.rfind("/") + 2]
                data = extract_data(graph, flip=True if side == "R" else False)
                label = 0
                fn, fp = [], []
                for name in data["names"]:
                    if name.startswith(self.pattern):
                        label = 1
                    fn.append(sum([1 for n in self.names_filter if name.startswith(n)]))
                    fp.append(1 if name.startswith(self.pattern) else 0)
                bck_filtered = np.asarray(data["bck"])[np.asarray(fn) == 1]
                spattern = np.asarray(data["bck"])[np.asarray(fp) == 1]
                dict_label[gfile] = label
                dict_bck[gfile] = data["bck"]
                dict_bck_filtered[gfile] = [list(p) for p in bck_filtered]
                dict_searched_pattern[gfile] = [list(p) for p in spattern]

            # save in parameters file
            param = {
                "dict_bck": dict_bck,
                "dict_bck_filtered": dict_bck_filtered,
                "dict_label": dict_label,
                "dict_searched_pattern": dict_searched_pattern,
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
            dict_searched_pattern = param["dict_searched_pattern"]
            dict_label = param["dict_label"]

        method = SVMPatternClassification(
            pattern=self.pattern,
            names_filter=self.names_filter,
            dict_bck=dict_bck,
            dict_bck_filtered=dict_bck_filtered,
            dict_searched_pattern=dict_searched_pattern,
            dict_label=dict_label,
        )

        # Inner cross validation - fix learning rate / momentum
        if self.step_2:
            print()
            print("---------------------------")
            print("---      STEP (2/3)     ---")
            print("--- FIX HYPERPARAMETERS ---")
            print("---------------------------")
            print()
            start = time.time()
            method.find_hyperparameters(self.graphs, self.param_file)
            end = time.time()
            print()
            print("STEP 2 took %s" % str(timedelta(seconds=int(end - start))))
            print()
        else:
            with open(self.param_file) as f:
                param = json.load(f)
            method.C = param["C"]
            method.gamma = param["gamma"]
            method.transrot_init = param["trans"]

        if self.step_3:
            print()
            print("--------------------------")
            print("---      STEP (3/3)    ---")
            print("--- SAVE TRAINED MODEL ---")
            print("--------------------------")
            print()
            start = time.time()
            method.learning(self.graphs)
            method.save(self.clf_file, self.scaler_file)
            end = time.time()
            print()
            print("STEP 3 took %s" % str(timedelta(seconds=int(end - start))))
            print()
