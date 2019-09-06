# -*- coding: utf-8 -*-
from __future__ import print_function
from ...deeptools.dataset import SulciDataset
from ...deeptools.early_stopping import EarlyStopping
from ...deeptools.models import UNet3D
from ..analyse.stats import esi_score

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import time
import copy
import json


class UnetSulciLabeling(object):
    ''' 3D U-Net for automatic sulci recognition

    Parameters
    ----------
    sulci_side_list : list of sulci names
    lr : learning rate
    momentum : momentume for SGD
    early_stopping : if True, early_stopping with patience=4 is used
    cuda : index of the GPU to use
    batch_size : number of sample per batch during learning
    data_augmentation : if True, random rotation of the images
    num_filter : number of init filter in the UNet (default=64)
    opt : optimizer to use ('Adam' or 'SGD')
    '''
    def __init__(self, sulci_side_list,
                 batch_size=3, cuda=0, lr=0.001, momentum=0.9,
                 num_filter=64, translation_file=None,
                 dict_bck2=None, dict_names=None):

        # training parameters
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.lr_range = [1e-2, 1e-3, 1e-4]
        self.momentum_range = [0.8, 0.7, 0.6, 0.5]

        # dataloader
        self.background = -1
        self.sulci_side_list = [str(s) for s in sulci_side_list]
        self.sslist = [ss for ss in self.sulci_side_list if not ss.startswith('unknown') and not ss.startswith('ventricle')]
        self.dict_sulci = {self.sulci_side_list[i]: i for i in range(len(self.sulci_side_list))}
        self.dict_num = {v: k for k, v in self.dict_sulci.items()}
        self.translation_file = translation_file

        # lazy memory
        self.dict_bck2 = {} if dict_bck2 is None else dict_bck2
        self.dict_names = {} if dict_names is None else dict_names

        # network
        self.num_filter = num_filter
        self.num_channel = 1
        self.pretrained_model_wts = None

        # device
        self.cuda = cuda
        if self.cuda is -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu", index=cuda)
        print('Working on', self.device)

    def learning(self, gfile_list_train, gfile_list_test):
        print('TRAINING ON %i + %i samples' %
              (len(gfile_list_train), len(gfile_list_test)))
        print()
        print('PARAMETERS')
        print('----------')
        print('batch_size:', self.batch_size)
        print('learning rate:', self.lr, 'momentum', self.momentum)
        print('number of filters:', self.num_filter)
        print()

        # DATASET / DATALOADERS
        print('Extract validation dataloader...')
        valdataset = SulciDataset(
            gfile_list_test, self.dict_sulci,
            train=False, translation_file=self.translation_file,
            dict_bck2=self.dict_bck2, dict_names=self.dict_names)
        valloader = torch.utils.data.DataLoader(
            valdataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0)

        print('Extract train dataloader...')
        traindataset = SulciDataset(
            gfile_list_train, self.dict_sulci,
            train=True, translation_file=self.translation_file,
            dict_bck2=self.dict_bck2, dict_names=self.dict_names)
        trainloader = torch.utils.data.DataLoader(
            traindataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0)

        # MODEL
        print('Network initialization...')
        model = UNet3D(self.num_channel, len(self.sulci_side_list),
                       final_sigmoid=False,
                       init_channel_number=self.num_filter)
        model = model.to(self.device)

        lr = self.lr
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=self.momentum, weight_decay=0)

        patience = 2
        divide_lr = EarlyStopping(patience=patience)
        es_stop = EarlyStopping(patience=patience*2)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        # TRAINING
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc, epoch_acc = 0., 0.
        best_epoch = 0
        num_epochs = 200
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            start_time = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()    # Set model to evaluate mode

                running_loss = 0.0

                # compute dataloader
                dataloader = trainloader if phase == 'train' else valloader

                # Iterate over data.
                y_pred, y_true = [], []
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forwards.det
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    y_pred.extend(preds[labels != self.background].tolist())
                    y_true.extend(labels[labels != self.background].tolist())

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = 1 - esi_score(
                    y_true, y_pred,
                    [self.dict_sulci[ss] for ss in self.sslist])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

            # early_stopping
            es_stop(epoch_loss, model)
            divide_lr(epoch_loss, model)

            if divide_lr.early_stop:
                print('Divide learning rate')
                lr = lr/2
                optimizer = optim.SGD(model.parameters(), lr=lr,
                                      momentum=self.momentum)
                divide_lr = EarlyStopping(patience=patience)

            if es_stop.early_stop:
                print("Early stopping")
                break

            print('Epoch took %i s.' % (time.time() - start_time))
            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}, Epoch {}'.format(best_acc, best_epoch))
        self.best_acc = best_acc
        self.best_epoch = best_epoch

        # load best model weights
        model.load_state_dict(best_model_wts)
        self.trained_model = model

    def labeling(self, gfile, bck2=None, names=None):
        print('Labeling', gfile)
        self.trained_model = self.trained_model.to(self.device)
        self.trained_model.eval()
        dict_bck2 = {} if bck2 is None else {gfile: bck2}
        dict_names = {} if names is None else {gfile: names}
        dataset = SulciDataset(
            [gfile], self.dict_sulci, train=False,
            translation_file=self.translation_file,
            dict_bck2=dict_bck2, dict_names=dict_names)
        data = dataset[0]

        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.unsqueeze(0)
            inputs = inputs.to(self.device)
            outputs = self.trained_model(inputs)
            if bck2 is None:
                bck2 = np.transpose(np.where(np.asarray(labels) != self.background))
            bck_T = np.transpose(bck2)
            _, preds = torch.max(outputs.data, 1)
            ypred = preds[0][bck_T[0], bck_T[1], bck_T[2]].tolist()
            ytrue = labels[bck_T[0], bck_T[1], bck_T[2]].tolist()
            yscores = outputs[0][:, bck_T[0], bck_T[1], bck_T[2]].tolist()
            yscores = np.transpose(yscores)
        return ytrue, ypred, yscores

    def find_hyperparameters(self, result_matrix, param_outfile, step=0):
        # STEP 0
        if step == 0:
            best_acc = 0
            for lr, result_list in zip(self.lr_range, result_matrix):
                # compute acc
                acc = []
                for result in result_list:
                    # TODO. verifier que les doublons ne posent pas de pb
                    # dans le calcul des sorties
                    acc.append(1 - esi_score(
                        result['true_label'], result['pred_label'],
                        self.sslist))
                print('lr: %f, acc: %f' % (lr, np.mean(acc)))
                # save best acc
                if np.mean(acc) > best_acc:
                    best_acc = np.mean(acc)
                    best_lr = lr
            param = {'best_lr0': best_lr,
                     'best_acc': best_acc,
                     'sulci_side_list': self.sulci_side_list}
            with open(param_outfile, 'w') as f:
                json.dump(param, f)
        # STEP 1
        elif step == 1:
            with open(param_outfile) as f:
                param = json.load(f)
            best_lr0 = param['best_lr0']
            best_lr = param['best_lr0']
            best_acc = param['best_acc']
            lr1_range = [best_lr0/4, best_lr0/2, best_lr0*2, best_lr0*4]
            for lr, result_list in zip(lr1_range, result_matrix):
                # compute acc
                acc = []
                for result in result_list:
                    acc.append(1 - esi_score(
                        result['true_label'], result['pred_label'],
                        self.sslist))
                print('lr: %f, acc: %f' % (lr, np.mean(acc)))
                # save best acc
                if np.mean(acc) > best_acc:
                    best_acc = np.mean(acc)
                    best_lr = lr
            param['best_lr1'] = best_lr
            param['best_acc'] = best_acc
            with open(param_outfile, 'w') as f:
                json.dump(param, f)
        # STEP 2
        elif step == 2:
            with open(param_outfile) as f:
                param = json.load(f)
            best_acc = param['best_acc']
            best_momentum = 0.9
            for momentum, result_list in zip(self.momentum_range,
                                             result_matrix):
                # compute acc
                acc = []
                for result in result_list:
                    acc.append(1 - esi_score(
                        result['true_label'], result['pred_label'],
                        self.sslist))
                print('momentum: %f, acc: %f' % (momentum, np.mean(acc)))
                # save best acc
                if np.mean(acc) > best_acc:
                    best_acc = np.mean(acc)
                    best_momentum = momentum
            param['best_momentum'] = best_momentum
            with open(param_outfile, 'w') as f:
                json.dump(param, f)

            # train with best parameters
            self.lr = param['best_lr1']
            self.momentum = param['best_momentum']

            print()
            print('Best hyperparameters:',
                  'learning rate %f, momentum %f, acc %f' %
                  (self.lr, self.momentum, param['best_acc']))
            print()

    def cv_inner(self, gfile_list_train, gfile_list_test,
                 param_outfile, step=0):

        # STEP 0
        if step == 0:
            momentum = 0.9
            result_matrix = []
            for lr in self.lr_range:
                print()
                print('TEST learning rate', lr)
                print('======================')
                result_list = self.test_hyperparameters(
                    lr, momentum, gfile_list_train, gfile_list_test)
                result_matrix.append(result_list)
        # STEP 1
        elif step == 1:
            with open(param_outfile) as f:
                param = json.load(f)
            momentum = 0.9
            best_lr0 = param['best_lr0']
            result_matrix = []
            for lr in [best_lr0/4, best_lr0/2, best_lr0*2, best_lr0*4]:
                print()
                print('TEST learning rate', lr)
                print('======================')
                result_list = self.test_hyperparameters(
                    lr, momentum, gfile_list_train, gfile_list_test)
                result_matrix.append(result_list)

        # STEP 2
        elif step == 2:
            with open(param_outfile) as f:
                param = json.load(f)
            best_lr1 = param['best_lr1']
            result_matrix = []
            for momentum in self.momentum_range:
                print()
                print('TEST momentum', momentum)
                print('======================')
                result_list = self.test_hyperparameters(
                    best_lr1, momentum, gfile_list_train, gfile_list_test)
                result_matrix.append(result_list)

        return result_matrix

    def test_hyperparameters(self, lr, momentum,
                             gfile_list_train, gfile_list_test):
        self.lr = lr
        self.momentum = momentum

        print()
        s = 'TRAIN WITH lr '+str(lr)+' momentum '+str(momentum)
        print(s)
        print('='*len(s))
        self.learning(gfile_list_train, gfile_list_test)

        # labeling
        print()
        print('TEST labeling')
        print('=============')
        result_list = []
        for gf in gfile_list_test:
            y_true, y_pred, _ = self.labeling(gf)
            result = pd.DataFrame()
            result['true_label'] = [self.dict_num[y] for y in y_true]
            result['pred_label'] = [self.dict_num[y] for y in y_pred]
            result_list.append(result)

        return result_list

    def load(self, model_file):
        self.trained_model = UNet3D(
            self.num_channel, len(self.sulci_side_list), final_sigmoid=False,
            init_channel_number=self.num_filter)
        self.trained_model.load_state_dict(torch.load(
            model_file, map_location='cpu'))
        self.trained_model = self.trained_model.to(self.device)
