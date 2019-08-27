# -*- coding: utf-8 -*-
from __future__ import print_function
from ...deeptools.dataset import PatternDataset ###
from ...deeptools.early_stopping import EarlyStopping
from ...deeptools.models import resnet18
from ..analyse.stats import bacc_score ###
from sklearn.model_selection import train_test_split

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
    def __init__(self, sulcus, cuda=None, names_filter=None,
                 lr=0.0001, momentum=0.9, batch_size=10):
        self.sulcus = sulcus
        self.names_filter = names_filter
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size

        self.lr_range = [1e-2, 1e-3, 1e-4, 1e-5]
        self.momentum_range = [0.8, 0.7, 0.6, 0.5]
        self.patience = 5
        self.division = 10

        if cuda == None:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu", index=cuda)
        print('Working on', self.device)

    def learning(self, data_train, data_test=None):
        print()
        print('LEARNING PARAMETERS')
        print('-------------------')
        print('batch_size:', self.batch_size)
        print('learning_rate:', self.lr, 'momentum', self.momentum)
        print('filter_fold:', self.fold_filter, 'ss_filter:', self.ss_filter)
        print('patience:', self.patience, 'division:', self.division)
        print()

        if data_test is None:
            data = data_train
            y = [sum([1 if n.startswith(self.sulcus) else 0 for n in set(np.array(np.load(os.path.join(
                d, 'names_2mm/%s.npy' % s)), dtype=np.str))]) for d, s in data_train]
            data_train, data_test = train_test_split(
                data, test_size=0.1, random_state=0,
                shuffle=True, stratify=y)
        data_val = data_test
        lr = self.lr

        # DATALOADERS
        print('Compute bounding box...')
        bb = np.array([[100, 0], [100, 0], [100, 0]])
        for d, s in data_train:
            bck = np.load(os.path.join(d, 'buckets_2mm/%s.npy' % s))
            if self.ss_filter is not None:
                labels = np.load(os.path.join(d, '%s_2mm/%s.npy' % (self.fold_filter, s)))
                labels = np.asarray(labels, dtype=np.str)
                bck = np.vstack([bck[[True if l.startswith(s) else False for l in labels]] for s in self.ss_filter])
            if len(bck) != 0:
                bb[:, 1] = np.max([np.max(bck, axis=0), bb[:, 1]], axis=0)
                bb[:, 0] = np.min([np.min(bck, axis=0), bb[:, 0]], axis=0)
        self.bb = bb
        
        if self.fold_filter == 'names':
            print('Compute initialization mask...')
            print('amask_file:', None)
            labels_filter = None
        else:
            print('Compute labels filter...')
            print('labels_filter:', self.ss_filter)
            labels_filter = self.ss_filter

        print('Create dataloader...')
        self.dataset_extractor = PatternDataset(self.sulcus,
            bb=bb, batch_size=self.batch_size,
            labels_filter=labels_filter)
        trainloader = self.dataset_extractor.extract_trainset(data_train)
        valloader = self.dataset_extractor.extract_testset(data_val)

        print('3D ResNet18')
        model = resnet18()
        model = model.to(self.device)
        print('SGD with nesterov momentum')
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=self.momentum, nesterov=True)
        y = [sum([1 if n.startswith(self.sulcus) else 0 for n in set(np.array(np.load(os.path.join(
            d, 'names_2mm/%s.npy' % s)), dtype=np.str))]) for d, s in data_train]
        class_sample_count = np.array(
            [len(np.where(y == t)[0]) for t in np.unique(y)])
        w = 1. / class_sample_count
        w = torch.tensor([w[0], w[1]], dtype=torch.float)
        print('CrossEntropyLoss weights', w)
        self.criterion = nn.CrossEntropyLoss(weight=w.to(self.device))

        # early stopping
        num_epochs = 200
        divide_lr = EarlyStopping(patience=self.patience)
        es_stop = EarlyStopping(patience=self.patience*2)

        # TRAINING
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = 100
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
                epoch_acc = bacc_score(y_true, y_pred, [0,1])

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
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        self.trained_model = model

    def labelisation(self, data, result_file):
        self.dataset_extractor = PatternDataset(self.sulcus,
            bb=self.bb, batch_size=self.batch_size,
            labels_filter=None)
        testloader = self.dataset_extractor.extract_testset(data)
        result = pd.DataFrame(columns=['y_true', 'y_pred'],
                              index=[s for d, s in data])
        with torch.no_grad():
            i = 0
            for data_test in testloader:
                print('Labelisation (%i/%i)' %(i, len(testloader)))
                start_time = time.time()
                inputs, labels = data_test
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.trained_model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # statistics
                slist = np.asarray(data)[:, 1][i*self.batch_size:(i+1)*self.batch_size]
                result.loc[slist, 'y_true'] = labels.tolist()
                result.loc[slist, 'y_pred'] = preds.tolist()
                result.loc[slist, 'score_0'] = np.array(outputs.data.tolist())[:,0]
                result.loc[slist, 'score_1'] = np.array(outputs.data.tolist())[:, 1]
                i += 1
                print('took %i s.' % (time.time()-start_time))
        result.to_csv(result_file)
    
    def labelisation_from_gfilename(self, gfilelist, result_file,
                                    translation_2mm):
        self.dataset_extractor = PatternDataset(self.sulcus,
            bb=self.bb, batch_size=self.batch_size,
            amask=None, labels_filter=None,
            gfile=True, translation_2mm=translation_2mm)
        testloader = self.dataset_extractor.extract_testset(gfilelist)
        result = pd.DataFrame(index=[s for s in gfilelist])
        with torch.no_grad():
            i = 0
            for data in testloader:
                print('Labelisation (%i/%i)' %(i, len(testloader)))
                start_time = time.time()
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.trained_model(inputs)
                _, preds = torch.max(outputs.data, 1)
                # statistics
                slist = gfilelist[i*self.batch_size:(i+1)*self.batch_size]
                result.loc[slist, 'y_pred'] = preds.tolist()
                result.loc[slist, 'score_0'] = np.array(outputs.data.tolist())[:,0]
                result.loc[slist, 'score_1'] = np.array(outputs.data.tolist())[:, 1]
                print('took %i s.' % (time.time()-start_time))
                i += 1
        result.to_csv(result_file)

    def find_hyperparameters(self, cvo_path, step=0):
        print()
        print('** FIND HYPERPARAMETERS')
        #### step 0
        if step == 0:
            best_bacc = 0
            it  = 0
            for lr in self.lr_range:
                # compute acc
                y_true, y_pred = [], []
                for cvi in range(3):
                    cvi_path = os.path.join(cvo_path, 'cv_%i' % cvi)
                    r = pd.read_csv(os.path.join(
                        cvi_path, 'result_%i%i.csv' % (step, it)), index_col=0)
                    y_true.extend([y for y in r['y_true']])
                    y_pred.extend([y for y in r['y_pred']])
                bacc = bacc_score(y_true, y_pred, [0, 1])
                print('lr: %f, bacc: %f' % (lr, bacc))
                if best_bacc < bacc:
                    best_bacc = bacc
                    best_lr = lr
                it += 1
            param = {'best_lr0': best_lr,
                     'best_bacc': best_bacc}
            with open(os.path.join(cvo_path, 'parameters.txt'), 'w') as f:
                json.dump(param, f)
        ### step 1
        elif step == 1:
            with open(os.path.join(cvo_path, 'parameters.txt')) as f:
                param = json.load(f)
            it = 0
            best_lr0 = param['best_lr0']
            best_lr = param['best_lr0']
            best_bacc = param['best_bacc']
            for lr in [best_lr0/4, best_lr0/2, best_lr0*2, best_lr0*4]:
                # compute acc
                y_true, y_pred = [], []
                for cvi in range(3):
                    cvi_path = os.path.join(cvo_path, 'cv_%i' % cvi)
                    r = pd.read_csv(os.path.join(
                        cvi_path, 'result_%i%i.csv' % (step, it)), index_col=0)
                    y_true.extend([y for y in r['y_true']])
                    y_pred.extend([y for y in r['y_pred']])
                bacc = bacc_score(y_true, y_pred, [0, 1])
                print('lr: %f, bacc: %f' % (lr, bacc))
                if best_bacc < bacc:
                    best_bacc = bacc
                    best_lr = lr
                it += 1
            param['best_lr1'] = best_lr
            param['best_bacc'] = best_bacc
            with open(os.path.join(cvo_path, 'parameters.txt'), 'w') as f:
                json.dump(param, f)
        #### step 2
        elif step == 2:
            with open(os.path.join(cvo_path, 'parameters.txt')) as f:
                param = json.load(f)
            it = 0
            best_bacc = param['best_bacc']
            best_momentum = 0.9
            for momentum in self.momentum_range:
                # compute acc
                y_true, y_pred = [], []
                for cvi in range(3):
                    cvi_path = os.path.join(cvo_path, 'cv_%i' % cvi)
                    r = pd.read_csv(os.path.join(
                        cvi_path, 'result_%i%i.csv' % (step, it)), index_col=0)
                    y_true.extend([y for y in r['y_true']])
                    y_pred.extend([y for y in r['y_pred']])
                bacc = bacc_score(y_true, y_pred, [0, 1])
                print('momentum: %f, bacc: %f' % (momentum, bacc))
                if best_bacc < bacc:
                    best_bacc = bacc
                    best_momentum = momentum
                it += 1
            param['best_momentum'] = best_momentum
            with open(os.path.join(cvo_path, 'parameters.txt'), 'w') as f:
                json.dump(param, f)

            # train with best parameters
            self.lr = param['best_lr1']
            self.momentum = param['best_momentum']

            print()
            print('Best hyperparameters: learning rate %f, momentum %f, acc %f' %
                  (self.lr, self.momentum, param['best_bacc']))
            print()

        elif step == 3:
            with open(os.path.join(cvo_path, 'parameters.txt')) as f:
                param = json.load(f)
            it = 0
            best_bacc = param['best_bacc']
            best_mask_dir = self.mask_dir_range[0]
            for mask_dir in self.mask_dir_range[1:]:
                # compute acc
                y_true, y_pred = [], []
                for cvi in range(3):
                    cvi_path = os.path.join(cvo_path, 'cv_%i' % cvi)
                    r = pd.read_csv(os.path.join(
                        cvi_path, 'result_%i%i.csv' % (step, it)), index_col=0)
                    y_true.extend([y for y in r['y_true']])
                    y_pred.extend([y for y in r['y_pred']])
                bacc = bacc_score(y_true, y_pred, [0, 1])
                print('momentum: %f, bacc: %f' % (momentum, bacc))
                if best_bacc < bacc:
                    best_bacc = bacc
                    best_mask_dir = mask_dir
                it += 1
            param['best_mask_dir'] = best_mask_dir
            with open(os.path.join(cvo_path, 'parameters.txt'), 'w') as f:
                json.dump(param, f)

            # train with best parameters
            self.lr = param['best_lr1']
            self.momentum = param['best_momentum']

            print()
            print('Best hyperparameters: learning rate %f, momentum %f, acc %f' %
                  (self.lr, self.momentum, param['best_bacc']))
            print()

    def cv_inner(self, data_train, data_test,
                 cvi_path, cvo_path, step=0):
        print()
        print('** CV INNER')
        #### step 0
        if step == 0:
            mask_dir = self.mask_dir_range[0]
            momentum = 0.9
            i = 0
            for lr in self.lr_range:
                self.test_hyperparameters(
                    lr, momentum, mask_dir, step, i, data_train, data_test,
                    cvi_path)
                i += 1
        ### step 1
        elif step == 1:
            with open(os.path.join(cvo_path, 'parameters.txt')) as f:
                param = json.load(f)
            mask_dir = self.mask_dir_range[0]
            momentum = 0.9
            i = 0
            best_lr0 = param['best_lr0']
            for lr in [best_lr0/4, best_lr0/2, best_lr0*2, best_lr0*4]:
                self.test_hyperparameters(
                    lr, momentum, mask_dir, step, i, data_train, data_test,
                    cvi_path)
                i += 1

        #### step 2
        elif step == 2:
            with open(os.path.join(cvo_path, 'parameters.txt')) as f:
                param = json.load(f)
            i = 0
            mask_dir = self.mask_dir_range[0]
            best_lr1 = param['best_lr1']
            for momentum in self.momentum_range:
                self.test_hyperparameters(
                    best_lr1, momentum, mask_dir, step, i,
                    data_train, data_test,
                    cvi_path)
                i += 1

        #### step 3
        elif step == 3:
            with open(os.path.join(cvo_path, 'parameters.txt')) as f:
                param = json.load(f)
            i = 0
            best_lr1 = param['best_lr1']
            best_momentum = param['best_momentum']
            for mask_dir in self.mask_dir_range:
                self.test_hyperparameters(
                    best_lr1, best_momentum, mask_dir, step, i,
                    data_train, data_test,
                    cvi_path)
                i += 1

    def test_hyperparameters(self, lr, momentum, mask_dir, step, it,
                             data_train, data_test, cvi_path):
        self.lr = lr
        self.momentum = momentum
        self.trained_model = None
        result_file = os.path.join(
            cvi_path, 'result_%i%i.csv' % (step, it))
        self.learning(data_train, data_test)
        self.labelisation(data_test, result_file)

    def load(self, model_file):
        self.trained_model = resnet18()
        self.trained_model.load_state_dict(torch.load(model_file, map_location='cpu'))
        self.trained_model = self.trained_model.to(self.device)
