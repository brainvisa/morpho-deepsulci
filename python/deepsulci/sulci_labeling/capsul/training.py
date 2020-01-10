'''
UNET Training Module
'''

from __future__ import print_function
from ...deeptools.dataset import extract_data
from ..method.unet import UnetSulciLabeling
from ..method.cutting import cutting
from ..analyse.stats import esi_score
from sklearn.model_selection import KFold, train_test_split
from capsul.api import Process
from soma import aims
from datetime import timedelta
import traits.api as traits
import numpy as np
import pandas as pd
import json
import torch
import sigraph
import os
import time
import six


class SulciDeepTraining(Process):
    '''
    Process to train a UNET neural network to automatically label the sulci.

    This process consists of four steps. Each step depends on the previous
    step (except step 4 that is independent of step 3).
    However, they can be started independently if the previous steps have
    already been completed.

    The first step is to extract from the graphs the data useful for training
    the neural network (buckets and corresponding sulcus names).
    These data are stored in Jason files (buckets and names in
    traindata_file and sulci list in param_file).

    The second step allows to set the hyperparameters (learning rate and
    momentum) by 3-fold cross-validation.
    These hyperparameters are saved in the Jason file param_file.

    The third step is to train the UNET neural network on the entire database.
    The neural network parameters are saved in the file model_param.mdsm

    The fourth step allows to set the cutting hyperparameter (threshold on the
    Calinski-Harabaz index) by 3-fold cross-validation.
    This hyperparameter is saved in the Jason file param_file.

    The model takes approximately 20 hours to be trained on the GPU with a
    training database of about 60 subjects (step 1: 15min, step 2: 16h, step 3:
    20min, step 4: 3h).

    **Warning:** Graphs should be of the same side!

    '''

    def __init__(self):
        super(SulciDeepTraining, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False),
                                             desc='training base graphs'))
        self.add_trait('graphs_notcut', traits.List(
            traits.File(output=False),
            desc='training base graphs before manual cutting of the'
                 ' elementary folds'))
        self.add_trait('cuda', traits.Int(
            output=False, desc='device on which to run the training'
                               '(-1 for cpu, i>=0 for the i-th gpu)'))
        self.add_trait('translation_file', traits.File(
            output=False, optional=True,
            desc='file (.trl) containing the translation of the sulci to'
                 'applied on the training base graphs (optional)'))
        self.add_trait('step_1', traits.Bool(
            True, output=False, optional=True,
            desc='perform the data extraction step from the graphs'))
        self.add_trait('step_2', traits.Bool(
            True, output=False, optional=True,
            desc='perform the hyperparameter tuning step'
                 ' (learning rate and momentum)'))
        self.add_trait('step_3', traits.Bool(
            True, output=False, optional=True,
            desc='perform the model training step'))
        self.add_trait('step_4', traits.Bool(
            True, output=False, optional=True,
            desc='perform the cutting hyperparameter tuning step'))

        self.add_trait('model_file', traits.File(
            output=True,
            desc='file (.mdsm) storing neural network parameters'))
        self.add_trait('param_file', traits.File(
            output=True, desc='file (.json) storing the hyperparameters'
                              ' (learning rate, momentum, cutting threshold)'))
        self.add_trait('traindata_file', traits.File(
            output=True, desc='file (.json) storing the data extracted'
                              ' from the training base graphs'))

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
            print()
            print('--------------------------------')
            print('---         STEP (1/4)       ---')
            print('--- EXTRACT DATA FROM GRAPHS ---')
            print('--------------------------------')
            print()
            start = time.time()

            sulci_side_list = set()
            dict_bck2, dict_names = {}, {}
            for gfile in self.graphs:
                graph = aims.read(gfile)
                if trfile is not None:
                    flt.translate(graph)
                data = extract_data(graph)
                dict_bck2[gfile] = data['bck2']
                dict_names[gfile] = data['names']
                for n in data['names']:
                    sulci_side_list.add(n)
            sulci_side_list = sorted(list(sulci_side_list))

            if os.path.exists(self.param_file):
                with open(self.param_file) as f:
                    param = json.load(f)
            else:
                param = {}
            param['sulci_side_list'] = [str(s) for s in sulci_side_list]
            with open(self.param_file, 'w') as f:
                json.dump(param, f)

            traindata = {}
            traindata['dict_bck2'] = dict_bck2
            traindata['dict_names'] = dict_names
            with open(self.traindata_file, 'w') as f:
                json.dump(traindata, f)
            end = time.time()
            print()
            print("STEP 1 took %s" % str(timedelta(seconds=int(end-start))))
            print()
        else:
            with open(self.param_file) as f:
                param = json.load(f)
            sulci_side_list = param['sulci_side_list']

            with open(self.traindata_file) as f:
                traindata = json.load(f)
            dict_bck2 = traindata['dict_bck2']
            dict_names = traindata['dict_names']

        # init method
        sslist = [ss for ss in sulci_side_list if not ss.startswith('unknown') and not ss.startswith('ventricle')]
        method = UnetSulciLabeling(
            sulci_side_list, num_filter=64, batch_size=1, cuda=self.cuda,
            translation_file=trfile,
            dict_bck2=dict_bck2, dict_names=dict_names)

        # Inner cross validation - fix learning rate / momentum
        if self.step_2:
            print()
            print('------------------------------------')
            print('---           STEP (2/4)         ---')
            print('--- FIX LEARNING RATE / MOMENTUM ---')
            print('------------------------------------')
            print()
            start = time.time()
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
            end = time.time()
            print()
            print("STEP 2 took %s" % str(timedelta(seconds=int(end-start))))
            print()
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
            start = time.time()
            method.trained_model = None
            gfile_list_train, gfile_list_test = train_test_split(
                self.graphs, test_size=0.1)
            method.learning(gfile_list_train, gfile_list_test)

            cpu_model = method.trained_model.to(torch.device('cpu'))
            torch.save(cpu_model.state_dict(), self.model_file)

            end = time.time()
            print()
            print("STEP 3 took %s" % str(timedelta(seconds=int(end-start))))
            print()

        # Inner cross validation - fix cutting threshold
        if self.step_4:
            print()
            print('-----------------------------')
            print('---       STEP (4/4)      ---')
            print('--- FIX CUTTING THRESHOLD ---')
            print('-----------------------------')
            print()
            start = time.time()

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
                    # extract data
                    graph = aims.read(gfile)
                    if trfile is not None:
                        flt.translate(graph)
                    data = extract_data(graph)
                    nbck = np.asarray(data['nbck'])
                    bck2 = np.asarray(data['bck2'])
                    names = np.asarray(data['names'])

                    graph_notcut = aims.read(gfile_notcut)
                    if trfile is not None:
                        flt.translate(graph_notcut)
                    data_notcut = extract_data(graph_notcut)
                    nbck_notcut = np.asarray(data_notcut['nbck'])
                    vert_notcut = np.asarray(data_notcut['vert'])

                    # compute labeling
                    _, _, yscores = method.labeling(gfile, bck2, names)

                    # organize dataframes
                    df = pd.DataFrame()
                    df['point_x'] = nbck[:, 0]
                    df['point_y'] = nbck[:, 1]
                    df['point_z'] = nbck[:, 2]
                    df.sort_values(by=['point_x', 'point_y', 'point_z'],
                                   inplace=True)

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
                                names, ypred_cut, sslist))*100)
                cvi += 1

            dict_mscores = {k: np.mean(v)
                            for k, v in six.iteritems(dict_scores)}

            print()
            for k, v in six.iteritems(dict_mscores):
                print('threshold: %f, accuracy mean: %f' % (k, v))
            param['cutting_threshold'] = max(dict_mscores,
                                             key=dict_mscores.get)
            print()
            print('Best threshold:', param['cutting_threshold'])

            with open(self.param_file, 'w') as f:
                json.dump(param, f)

            end = time.time()
            print()
            print("STEP 4 took %s" % str(timedelta(seconds=int(end-start))))
            print()
