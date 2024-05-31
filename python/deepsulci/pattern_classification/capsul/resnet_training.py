"""
ResNet Training Module
"""

from ...deeptools.dataset import extract_data
from ..method.resnet import ResnetPatternClassification
from sklearn.model_selection import StratifiedKFold, train_test_split
from capsul.api import Process
from soma import aims
from datetime import timedelta

from soma.controller import File, field
import numpy as np
import json
import torch
import os
import time


class PatternDeepTraining(Process):
    """
    Process to train a 3D ResNet-18 to recognize a searched fold pattern.

    This process consists of three steps. Each step depends on the previous
    step. However, they can be started independently if the previous steps have
    already been completed.

    The first step is to extract from the graphs the data useful for training
    the neural networks (buckets, labels and bounding box of the names_filter).
    These data are stored in Jason files (buckets and labels in
    traindata_file and bounding box in param_file).

    The second step allows to set the hyperparameters (learning rate and
    momentum) by 3-fold cross-validation.These hyperparameters are saved in the
    Jason file param_file.

    The third step is to train the 18-layer 3D ResNet neural network on the
    data. The neural network parameters are saved in the file model_param.mdsm

    The model takes approximately 3 hours to be trained on the GPU with a
    training database of 100 subjects (step 1: 15min, step 2: 2h30, step 3:
    10min).

    **Warning:** The searched pattern must have been manually labeled on the
    graphs of the training database containing it.

    """

    graphs: field(type_=list[File], doc="training base graphs")
    pattern: field(type_=str, doc="vertex name representing the searched pattern")
    names_filter: field(
        type_=list[str],
        doc="list of vertex names defining the bounding box"
        " used as input for the neural network",
    )
    batch_size: field(type_=int, doc="batch size used to train the neural network") = 10
    cuda: field(
        type_=int,
        doc="device on which to run the training" "(-1 for cpu, i>=0 for the i-th gpu)",
    )
    step_1: field(
        type_=bool,
        optional=True,
        doc="perform the data extraction step from the graphs",
    ) = True
    step_2: field(
        type_=bool,
        optional=True,
        doc="perform the hyperparameter tuning step (learning rate and momentum)",
    ) = True
    step_3: field(
        type_=bool,
        optional=True,
        doc="perform the model training step",
    ) = True
    model_file: field(
        type_=File, write=True, doc="file (.mdsm) storing neural network parameters"
    )
    param_file: field(
        type_=File,
        write=True,
        doc="file (.json) storing the hyperparameters"
        " (bounding_box, learning rate and momentum)",
    )
    traindata_file: field(
        type_=File,
        write=True,
        doc="file (.json) storing the data extracted from the training base graphs",
    )

    def execution(self, context):
        agraphs = np.array(self.graphs)

        # Compute bounding box and labels
        if self.step_1:
            print()
            print("--------------------------------")
            print("---         STEP (1/3)       ---")
            print("--- EXTRACT DATA FROM GRAPHS ---")
            print("--------------------------------")
            print()
            start = time.time()

            bb = np.array([[100, -100], [100, -100], [100, -100]])
            dict_label, dict_bck = {}, {}
            for gfile in self.graphs:
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
                if len(bck_filtered) != 0:
                    bb[:, 1] = np.max([np.max(bck_filtered, axis=0), bb[:, 1]], axis=0)
                    bb[:, 0] = np.min([np.min(bck_filtered, axis=0), bb[:, 0]], axis=0)

            # save in parameters file
            if os.path.exists(self.param_file):
                with open(self.param_file) as f:
                    param = json.load(f)
            else:
                param = {}
            traindata = {}
            param["bounding_box"] = [list(b) for b in bb]
            traindata["dict_bck"] = dict_bck
            traindata["dict_label"] = dict_label
            with open(self.traindata_file, "w") as f:
                json.dump(traindata, f)
            with open(self.param_file, "w") as f:
                json.dump(param, f)
            end = time.time()
            print()
            print("STEP 1 took %s" % str(timedelta(seconds=int(end - start))))
            print()
        else:
            with open(self.param_file) as f:
                param = json.load(f)
            with open(self.traindata_file) as f:
                traindata = json.load(f)
            bb = np.asarray(param["bounding_box"])
            dict_bck = traindata["dict_bck"]
            dict_label = traindata["dict_label"]

        method = ResnetPatternClassification(
            bb,
            pattern=self.pattern,
            cuda=self.cuda,
            names_filter=self.names_filter,
            dict_bck=dict_bck,
            dict_label=dict_label,
        )

        # Inner cross validation - fix learning rate / momentum
        if self.step_2:
            print()
            print("------------------------------------")
            print("---           STEP (2/3)         ---")
            print("--- FIX LEARNING RATE / MOMENTUM ---")
            print("------------------------------------")
            print()
            start = time.time()
            y = np.asarray([dict_label[g] for g in self.graphs])
            n_cvinner = 3
            print("RANDOM STATE 0")
            skf = StratifiedKFold(n_splits=n_cvinner, shuffle=True, random_state=0)
            for step in range(3):
                print()
                print("**** STEP (%i/3) ****" % step)
                print()
                result_matrix = []
                cvi = 1
                for train, test in skf.split(self.graphs, y):
                    print()
                    print("** CV (%i/3) **" % cvi)
                    print()
                    glist_train = agraphs[train]
                    glist_test = agraphs[test]
                    result_list = method.cv_inner(
                        glist_train,
                        glist_test,
                        y[train],
                        y[test],
                        self.param_file,
                        step,
                    )
                    result_matrix.append(result_list)
                    cvi += 1

                print()
                print("** FIND HYPERPARAMETERS **")
                print()
                method.find_hyperparameters(result_matrix, self.param_file, step)
            end = time.time()
            print()
            print("STEP 2 took %s" % str(timedelta(seconds=int(end - start))))
            print()
        else:
            with open(self.param_file) as f:
                param = json.load(f)
            method.lr = param["best_lr1"]
            method.momentum = param["best_momentum"]

        # Train deep model
        if self.step_3:
            print()
            print("------------------------")
            print("---    STEP (3/3)    ---")
            print("--- TRAIN DEEP MODEL ---")
            print("------------------------")
            print()
            start = time.time()
            method.trained_model = None
            y = np.asarray([dict_label[g] for g in self.graphs])
            gfile_list_train, gfile_list_test = train_test_split(
                self.graphs, test_size=0.1, stratify=y
            )
            y_train = np.asarray([dict_label[g] for g in gfile_list_train])
            y_test = np.asarray([dict_label[g] for g in gfile_list_test])
            method.learning(gfile_list_train, gfile_list_test, y_train, y_test)

            cpu_model = method.trained_model.to(torch.device("cpu"))
            torch.save(cpu_model.state_dict(), self.model_file)
            end = time.time()
            print()
            print("STEP 3 took %s" % str(timedelta(seconds=int(end - start))))
            print()
