from __future__ import print_function
from ...deeptools.dataset import extract_data
from ..method.resnet import ResnetPatternClassification
from sklearn.model_selection import StratifiedKFold, train_test_split
from capsul.api import Process
from soma import aims

import traits.api as traits
import numpy as np
import json
import torch
import os


class PatternDeepTraining(Process):
    def __init__(self):
        super(PatternDeepTraining, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False)))
        self.add_trait('pattern', traits.Str(output=False))
        self.add_trait('names_filter', traits.ListStr(output=False))
        self.add_trait('batch_size', traits.Int(
            10, output=False, optional=True))
        self.add_trait('cuda', traits.Int(output=False))
        self.add_trait('step_1', traits.Bool(
            True, output=False, optional=True))
        self.add_trait('step_2', traits.Bool(
            True, output=False, optional=True))
        self.add_trait('step_3', traits.Bool(
            True, output=False, optional=True))

        self.add_trait('model_file', traits.File(output=True))
        self.add_trait('param_file', traits.File(output=True))
        self.add_trait('traindata_file', traits.File(output=True))

    def _run_process(self):
        agraphs = np.array(self.graphs)

        # Compute bounding box and labels
        if self.step_1:
            print()
            print('--------------------------------')
            print('---         STEP (1/3)       ---')
            print('--- EXTRACT DATA FROM GRAPHS ---')
            print('--------------------------------')
            print()

            bb = np.array([[100, -100], [100, -100], [100, -100]])
            dict_label, dict_bck = {}, {}
            for gfile in self.graphs:
                graph = aims.read(gfile)
                side = gfile[gfile.rfind('/')+1:gfile.rfind('/')+2]
                data = extract_data(graph, flip=True if side == 'R' else False)
                label = 0
                fn = []
                for name in data['names']:
                    if name.startswith(self.pattern):
                        label = 1
                    fn.append(sum([1 for n in self.names_filter if name.startswith(n)]))
                bck_filtered = np.asarray(data['bck2'])[np.asarray(fn) == 1]
                dict_label[gfile] = label
                dict_bck[gfile] = data['bck2']
                if len(bck_filtered) != 0:
                    bb[:, 1] = np.max([np.max(bck_filtered, axis=0), bb[:, 1]],
                                      axis=0)
                    bb[:, 0] = np.min([np.min(bck_filtered, axis=0), bb[:, 0]],
                                      axis=0)

            # save in parameters file
            if os.path.exists(self.param_file):
                with open(self.param_file) as f:
                    param = json.load(f)
            else:
                param = {}
            traindata = {}
            param['bounding_box'] = [list(b) for b in bb]
            traindata['dict_bck'] = dict_bck
            traindata['dict_label'] = dict_label
            with open(self.traindata_file, 'w') as f:
                json.dump(traindata, f)
            with open(self.param_file, 'w') as f:
                json.dump(param, f)
        else:
            with open(self.param_file) as f:
                param = json.load(f)
            with open(self.traindata_file) as f:
                traindata = json.load(f)
            bb = np.asarray(param['bounding_box'])
            dict_bck = traindata['dict_bck']
            dict_label = traindata['dict_label']

        method = ResnetPatternClassification(
            bb, pattern=self.pattern, cuda=self.cuda,
            names_filter=self.names_filter,
            dict_bck=dict_bck, dict_label=dict_label)

        # Inner cross validation - fix learning rate / momentum
        if self.step_2:
            print()
            print('------------------------------------')
            print('---           STEP (2/3)         ---')
            print('--- FIX LEARNING RATE / MOMENTUM ---')
            print('------------------------------------')
            print()
            y = np.asarray([dict_label[g] for g in self.graphs])
            n_cvinner = 3
            skf = StratifiedKFold(n_splits=n_cvinner, shuffle=True,
                                  random_state=0)
            for step in range(3):
                print()
                print('**** STEP (%i/3) ****' % step)
                print()
                result_matrix = []
                cvi = 1
                for train, test in skf.split(self.graphs, y):
                    print()
                    print('** CV (%i/3) **' % cvi)
                    print()
                    glist_train = agraphs[train]
                    glist_test = agraphs[test]
                    result_list = method.cv_inner(
                        glist_train, glist_test, y[train], y[test],
                        self.param_file, step)
                    result_matrix.append(result_list)
                    cvi += 1

                print()
                print('** FIND HYPERPARAMETERS **')
                print()
                method.find_hyperparameters(
                    result_matrix, self.param_file, step)
        else:
            with open(self.param_file) as f:
                param = json.load(f)
            method.lr = param['best_lr1']
            method.momentum = param['best_momentum']

        # Train deep model
        if self.step_3:
            print()
            print('------------------------')
            print('---    STEP (3/3)    ---')
            print('--- TRAIN DEEP MODEL ---')
            print('------------------------')
            print()
            method.trained_model = None
            y = np.asarray([dict_label[g] for g in self.graphs])
            gfile_list_train, gfile_list_test = train_test_split(
                self.graphs, test_size=0.1, stratify=y)
            y_train = np.asarray([dict_label[g] for g in gfile_list_train])
            y_test = np.asarray([dict_label[g] for g in gfile_list_test])
            method.learning(gfile_list_train, gfile_list_test,
                            y_train, y_test)

            cpu_model = method.trained_model.to(torch.device('cpu'))
            torch.save(cpu_model.state_dict(), self.model_file)
