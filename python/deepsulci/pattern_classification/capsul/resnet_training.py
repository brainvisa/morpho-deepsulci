from __future__ import print_function
from ..method.resnet import ResnetPatternClassification
from sklearn.model_selection import KFold, train_test_split
from capsul.api import Process
import traits.api as traits
import numpy as np
import json
import torch


class PatternDeepTraining(Process):
    def __init__(self):
        super(PatternDeepTraining, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False)))
        self.add_trait('pattern', traits.Str(output=False))
        self.add_trait('names_filter', traits.ListStr(output=False))
        self.add_trait('batch_size', traits.Int(
            output=False, optional=True, default=10))
        self.add_trait('cuda', traits.Int(output=False))
        self.add_trait('step_1', traits.Bool(
            output=False, optional=True, default=True))
        self.add_trait('step_2', traits.Bool(
            output=False, optional=True, default=True))

        self.add_trait('model_file', traits.File(output=True))
        self.add_trait('param_file', traits.File(output=True))

    def _run_process(self):
        agraphs = np.array(self.graphs)
        method = ResnetPatternClassification(
            self.graphs, self.pattern,
            cuda=self.cuda, names_filter=self.names_filter)

        # Inner cross validation - fix learning rate / momentum
        if self.step_1:
            print()
            print('------------------------------------')
            print('---           STEP (1/2)         ---')
            print('--- FIX LEARNING RATE / MOMENTUM ---')
            print('------------------------------------')
            print()
            n_cvinner = 3
            kf = KFold(n_splits=n_cvinner, shuffle=True, random_state=0)
            for step in range(3):
                print()
                print('**** STEP (%i/3) ****' % step)
                print()
                result_matrix = None
                cvi = 1
                for train, test in kf.split(self.graphs):
                    print()
                    print('** CV (%i/3) **' % cvi)
                    print()
                    glist_train = agraphs[train]
                    glist_test = agraphs[test]
                    result_m = method.cv_inner(
                        glist_train, glist_test, self.param_file, step)

                    if result_matrix is None:
                        result_matrix = result_m
                    else:
                        for i in range(n_cvinner):
                            result_matrix[i].extend(result_m[i])
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
        if self.step_2:
            print()
            print('------------------------')
            print('---    STEP (2/2)    ---')
            print('--- TRAIN DEEP MODEL ---')
            print('------------------------')
            print()
            method.trained_model = None
            gfile_list_train, gfile_list_test = train_test_split(
                self.graphs, test_size=0.1)
            method.learning(gfile_list_train, gfile_list_test)

            cpu_model = method.trained_model.to(torch.device('cpu'))
            torch.save(cpu_model.state_dict(), self.model_file)
