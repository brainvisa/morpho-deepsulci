from __future__ import print_function
from ..method.unet import UnetSulciLabeling
from ..method.cutting import cutting
from ..analyse.stats import esi_score
from sklearn.model_selection import KFold, train_test_split
from capsul.api import Process
from soma import aims
import traits.api as traits
import numpy as np
import pandas as pd
import json
import torch
import sigraph
import os


class SulciDeepTraining(Process):
    '''
    Graphs should be of the same side!
    '''
    def __init__(self):
        super(SulciDeepTraining, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False)))
        self.add_trait('graphs_notcut', traits.List(
            traits.File(output=False)))
        self.add_trait('cuda', traits.Int(output=False))
        self.add_trait('translation_file', traits.File(
            output=False, optional=True))
        self.add_trait('step_1', traits.Bool(
            output=False, optional=True, default=True))
        self.add_trait('step_2', traits.Bool(
            output=False, optional=True, default=True))
        self.add_trait('step_3', traits.Bool(
            output=False, optional=True, default=True))
        self.add_trait('step_4', traits.Bool(
            output=False, optional=True, default=True))

        self.add_trait('model_file', traits.File(output=True))
        self.add_trait('param_file', traits.File(output=True))

    def _run_process(self):
        agraphs = np.asarray(self.graphs)
        agraphs_notcut = np.asarray(self.graphs_notcut)
        if os.path.exists(self.translation_file):
            flt = sigraph.FoldLabelsTranslator()
            flt.readLabels(self.translation_file)
            trfile = self.translation_file
        else:
            trfile = None
            print('Translation file not found.')

        # compute sulci_side_list
        if self.step_1:
            print('-------------------------------')
            print('---        STEP (1/4)       ---')
            print('--- EXTRACT SULCI SIDE LIST ---')
            print('-------------------------------')
            print()
            sulci_side_list = set()
            for gfile in agraphs:
                graph = aims.read(gfile)
                if trfile is not None:
                    flt.translate(graph)
                for vertex in graph.vertices():
                    if 'name' in vertex:
                        sulci_side_list.add(vertex['name'])
            sulci_side_list = sorted(list(sulci_side_list))
            print('sulci side list:', sulci_side_list)

            with open(self.param_file) as f:
                param = json.load(f)
            param['sulci_side_list'] = [str(s) for s in sulci_side_list]
            with open(self.param_file, 'w') as f:
                json.dump(param, f)
        else:
            with open(self.param_file) as f:
                param = json.load(f)
            sulci_side_list = param['sulci_side_list']
        sslist = [ss for ss in sulci_side_list if not ss.startswith('unknown') and not ss.startswith('ventricle')]

        # init method
        method = UnetSulciLabeling(
            sulci_side_list, num_filter=64, batch_size=1, cuda=self.cuda,
            translation_file=trfile)

        # Inner cross validation - fix learning rate / momentum
        if self.step_2:
            print()
            print('------------------------------------')
            print('---           STEP (2/4)         ---')
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
        if self.step_3:
            print()
            print('------------------------')
            print('---    STEP (3/4)    ---')
            print('--- TRAIN DEEP MODEL ---')
            print('------------------------')
            print()
            method.trained_model = None
            gfile_list_train, gfile_list_test = train_test_split(
                self.graphs, test_size=0.1)
            method.learning(gfile_list_train, gfile_list_test)

            cpu_model = method.trained_model.to(torch.device('cpu'))
            torch.save(cpu_model.state_dict(), self.model_file)

        # Inner cross validation - fix cutting threshold
        if self.step_4:
            print()
            print('-----------------------------')
            print('---       STEP (4/4)      ---')
            print('--- FIX CUTTING THRESHOLD ---')
            print('-----------------------------')
            print()

            threshold_range = [50, 100, 150, 200, 250, 300]
            dict_scores = {th: [] for th in threshold_range}
            with open(self.param_file) as f:
                param = json.load(f)

            cvi = 0
            n_cvinner = 3
            kf = KFold(n_splits=n_cvinner, shuffle=True, random_state=0)
            for train, test in kf.split(self.graphs):
                print()
                print('**** CV (%i/3) ****' % cvi)
                print()
                glist_train = agraphs[train]
                glist_test = agraphs[test]
                glist_notcut_test = agraphs_notcut[test]

                # train model
                print()
                print('** TRAIN MODEL **')
                print()
                method.trained_model = None
                method.lr = param['best_lr1']
                method.momentum = param['best_momentum']
                method.learning(glist_train, glist_test)

                # test thresholds
                print()
                print('** TEST THRESHOLDS **')
                print()
                for gfile, gfile_notcut in zip(glist_test, glist_notcut_test):
                    ytrue, yscores, nbck, bck2 = method.labeling(
                        gfile, rytrue=True, ryscores=True,
                        rnbck=True, rbck2=True)
                    ytrue = [sulci_side_list[y] for y in ytrue]
                    df = pd.DataFrame()
                    nbck = np.asarray(nbck)
                    df['point_x'] = nbck[:, 0]
                    df['point_y'] = nbck[:, 1]
                    df['point_z'] = nbck[:, 2]
                    df.sort_values(by=['point_x', 'point_y', 'point_z'],
                                   inplace=True)
                    vert_notcut, nbck_notcut = method.labeling(
                        gfile_notcut, rvert=True, rnbck=True)
                    df_notcut = pd.DataFrame()
                    nbck_notcut = np.asarray(nbck_notcut)
                    df_notcut['vert'] = vert_notcut
                    df_notcut['point_x'] = nbck_notcut[:, 0]
                    df_notcut['point_y'] = nbck_notcut[:, 1]
                    df_notcut['point_z'] = nbck_notcut[:, 2]
                    df_notcut.sort_values(by=['point_x', 'point_y', 'point_z'],
                                          inplace=True)
                    if (len(df) != len(df_notcut)):
                        print()
                        print('ERROR no matches between %s and %s' % (
                            gfile, gfile_notcut))
                        print('--- Files ignored to fix the threshold')
                        print()
                    else:
                        df['vert_notcut'] = list(df_notcut['vert'])
                        df.sort_index(inplace=True)
                        for threshold in threshold_range:
                            ypred_cut = cutting(
                                yscores, df['vert_notcut'], bck2, threshold)
                            ypred_cut = [sulci_side_list[y] for y in ypred_cut]
                            dict_scores[threshold].append((1-esi_score(
                                ytrue, ypred_cut, sslist))*100)
                cvi += 1

            dict_mscores = {k: np.mean(v) for k, v in dict_scores.iteritems()}

            print()
            for k, v in dict_mscores.iteritems():
                print('threshold: %f, accuracy mean: %f' % (k, v))
            param['cutting_threshold'] = max(dict_mscores,
                                             key=dict_mscores.get)
            print()
            print('Best threshold:', param['cutting_threshold'])

            with open(self.param_file, 'w') as f:
                json.dump(param, f)
