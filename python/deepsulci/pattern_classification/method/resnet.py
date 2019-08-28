# -*- coding: utf-8 -*-
from __future__ import print_function
from ...deeptools.dataset import PatternDataset
from ...deeptools.early_stopping import EarlyStopping
from ...deeptools.models import resnet18
from ..analyse.stats import balanced_accuracy
from soma import aims

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import os
import time
import copy
import json


class ResnetPatternClassification:
    def __init__(self, pattern=None, cuda=-1, names_filter=None,
                 lr=0.0001, momentum=0.9, batch_size=10):
        self.pattern = pattern
        self.names_filter = names_filter
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size

        self.lr_range = [1e-2, 1e-3, 1e-4, 1e-5]
        self.momentum_range = [0.8, 0.7, 0.6, 0.5]
        self.patience = 5
        self.division = 10

        if cuda is -1:
            self.device = torch.device("cpu")
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
        print('patience:', self.patience, 'division:', self.division)
        print()

        # DATASET / DATALOADERS
        print('Compute bounding box...')
        bck_types = ['aims_ss', 'aims_bottom', 'aims_other']
        bb = np.array([[100, -100], [100, -100], [100, -100]])
        y = []
        for gfile in gfile_list_train:
            graph = aims.read(gfile)
            side = gfile[gfile.rfind('/')+1:gfile.rfind('/')+2]
            trans_tal = aims.GraphManip.talairach(graph)
            vs = graph['voxel_size']
            label = 0
            for vertex in graph.vertices():
                if 'name' in vertex:
                    name = vertex['name']
                    if name.startswith(self.pattern):
                        label = 1
                    filter = sum(
                        [1 if name.startswith(n) else 0 for n in self.names_filter])
                    if filter != 0:
                        for bck_type in bck_types:
                            if bck_type in vertex:
                                bucket = vertex[bck_type][0]
                                for point in bucket.keys():
                                    fpt = [p * v for p, v in zip(point, vs)]
                                    trans_pt = list(trans_tal.transform(fpt))
                                    if (side == 'R'):
                                        trans_pt[0] *= -1
                                    bb[:, 1] = np.max(
                                        [trans_pt, bb[:, 1]], axis=0)
                                    bb[:, 0] = np.min(
                                        [trans_pt, bb[:, 0]], axis=0)
            y.append(label)
        self.bb = [[int(round(x/2)) for x in p] for p in bb]

        print('Extract validation dataloader...')
        valdataset = PatternDataset(
            gfile_list_test, self.pattern, self.bb, train=False)
        valloader = torch.utils.data.DataLoader(
            valdataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0)

        print('Extract train dataloader...')
        traindataset = PatternDataset(
            gfile_list_train, self.pattern, self.bb, train=True)
        trainloader = torch.utils.data.DataLoader(
            traindataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0)

        # MODEL
        print('Network initialization...')
        model = resnet18()
        model = model.to(self.device)

        # OPTIMIZER
        lr = self.lr
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=self.momentum, nesterov=True)
        divide_lr = EarlyStopping(patience=self.patience)
        es_stop = EarlyStopping(patience=self.patience*2)

        # LOSS FUNCTION
        class_sample_count = np.array(
            [len(np.where(y == t)[0]) for t in np.unique(y)])
        w = 1. / class_sample_count
        w = torch.tensor([w[0], w[1]], dtype=torch.float)
        self.criterion = nn.CrossEntropyLoss(weight=w.to(self.device))

        # TRAINING
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        num_epochs = 200
        print()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            start_time = time.time()

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                    dataloader = trainloader
                else:
                    model.eval()   # Set model to evaluate mode
                    dataloader = valloader

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                y_pred, y_true = [], []
                for inputs, labels in dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
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
                    running_corrects += torch.sum(preds == labels.data)
                    y_true.extend(labels.tolist())
                    y_pred.extend(preds.tolist())

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = balanced_accuracy(y_true, y_pred, [0, 1])

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc >= best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            # early_stopping
            es_stop(epoch_loss, model)
            divide_lr(epoch_loss, model)

            if divide_lr.early_stop:
                print('Divide learning rate by', self.division)
                lr = lr/self.division
                optimizer = optim.SGD(model.parameters(), lr=lr,
                                      momentum=self.momentum)
                divide_lr = EarlyStopping(patience=self.patience)

            if es_stop.early_stop:
                print("Early stopping")
                break
            time_elapsed = time.time() - start_time
            print('Epoch took {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print()

        time_elapsed = time.time() - since
        print()
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        self.trained_model = model

    def labeling(self, gfile_list):
        self.trained_model = self.trained_model.to(self.device)
        self.trained_model.eval()

        dataset = PatternDataset(
            gfile_list, self.pattern, self.bb, train=False)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=0)

        result = pd.DataFrame(index=[s for s in gfile_list])
        start_time = time.time()
        with torch.no_grad():
            i = 0
            for data in dataloader:
                print('Labeling (%i/%i)' % (i, len(dataloader)))
                start_time = time.time()
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.trained_model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # statistics
                slist = gfile_list[i*self.batch_size:(i+1)*self.batch_size]
                result.loc[slist, 'y_pred'] = preds.tolist()
                result.loc[slist, 'y_true'] = labels.tolist()
                i += 1
        print('Labeling took %i s.' % (time.time()-start_time))

        return result

    def find_hyperparameters(self, result_matrix, param_outfile, step=0):
        # STEP 0
        if step == 0:
            best_acc = 0
            for lr, result_list in zip(self.lr_range, result_matrix):
                # compute acc
                acc = []
                for result in result_list:
                    acc.append(balanced_accuracy(
                        result['y_true'], result['y_pred'], [0, 1]))
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
                    acc.append(balanced_accuracy(
                        result['y_true'], result['y_pred'], [0, 1]))
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
                    acc.append(balanced_accuracy(
                        result['y_true'], result['y_pred'], [0, 1]))
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
                result = self.test_hyperparameters(
                    lr, momentum,
                    gfile_list_train, gfile_list_test)
                result_matrix.append(result)
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
                result = self.test_hyperparameters(
                    lr, momentum,
                    gfile_list_train, gfile_list_test)
                result_matrix.append(result)

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
                result = self.test_hyperparameters(
                    best_lr1, momentum,
                    gfile_list_train, gfile_list_test)
                result_matrix.append(result)

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

        print()
        print('TEST labeling')
        print('=============')
        result = self.labeling(gfile_list_test)

        return result

    def load(self, model_file):
        self.trained_model = resnet18()
        self.trained_model.load_state_dict(torch.load(
            model_file, map_location='cpu'))
        self.trained_model = self.trained_model.to(self.device)
