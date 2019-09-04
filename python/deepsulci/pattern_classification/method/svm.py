# -*- coding: utf-8 -*-
from __future__ import print_function
from ..analyse.stats import balanced_accuracy
from soma import aims
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.svm import SVC
from pcl import registration as reg

import numpy as np
import json
import itertools
import pcl


class SVMPatternClassification(object):
    def __init__(self, pattern=None, names_filter=None,
                 C=1, gamma=0.01, trans=[0],
                 dict_bck=None, dict_bck_filtered=None,
                 dict_searched_pattern=None, dict_label=None):
        self.pattern = pattern
        self.names_filter = names_filter
        self.C = C
        self.gamma = gamma
        self.transrot_init = []
        for x in trans:
            for y in trans:
                for z in trans:
                    self.transrot_init.append([[1, 0, 0, x],
                                               [0, 1, 0, y],
                                               [0, 0, 1, z],
                                               [0, 0, 0, 1]])

        self.C_range = [0.001, 0.01] #np.logspace(-4, -1, 4)
        self.gamma_range = [10, 100] #np.logspace(-1, 3, 5)
        self.trans_range = [[0],
                            [-5, 0, 5]] #,
#                            [-10, 0, 10],
#                            [-20, 0, 20],
#                            [-5, -10, 0, 10, 5],
#                            [-20, -10, 0, 10, -20]]

        if dict_bck is None:
            self.dict_bck = {}
        else:
            self.dict_bck = dict_bck
        if dict_bck_filtered is None:
            self.dict_bck_filtered = {}
        else:
            self.dict_bck_filtered = dict_bck_filtered
        if dict_searched_pattern is None:
            self.dict_searched_pattern = {}
        else:
            self.dict_searched_pattern = dict_searched_pattern
        if dict_label is None:
            self.dict_label = {}
        else:
            self.dict_label = dict_label

        self.bck_filtered_list = []
        self.label_list = []
        self.searched_pattern_list = []

    def learning(self, gfile_list):
        self.bck_filtered_list, self.label_list, self.distmap_list = [], [], []
        # Extract buckets and labels from the graphs
        bck_types = ['aims_ss', 'aims_bottom', 'aims_other']
        label = np.NaN if self.pattern is None else 0
        for gfile in gfile_list:
            if gfile not in self.dict_bck.keys():
                graph = aims.read(gfile)
                side = gfile[gfile.rfind('/')+1:gfile.rfind('/')+2]
                trans_tal = aims.GraphManip.talairach(graph)
                vs = graph['voxel_size']
                label = 0
                bck, bck_filtered, searched_pattern = [], [], []
                for vertex in graph.vertices():
                    if 'name' in vertex:
                        n = vertex['name']
                        if n.startswith(self.pattern):
                            label = 1
                        nf = sum([1 for m in self.nfilter if n.startswith(m)])
                        pf = 1 if n.startswith(self.pattern) else 0
                        for bck_type in bck_types:
                            if bck_type in vertex:
                                bucket = vertex[bck_type][0]
                                for point in bucket.keys():
                                    fpt = [p * v for p, v in zip(point, vs)]
                                    trpt = list(trans_tal.transform(fpt))
                                    if (side == 'R'):
                                        trpt[0] *= -1
                                    trpt_2mm = [int(round(p/2)) for p in trpt]
                                    if trpt_2mm not in bck:
                                        bck.append(trpt_2mm)
                                        if nf:
                                            bck_filtered.append(trpt_2mm)
                                        if pf:
                                            searched_pattern.append(trpt_2mm)

                # save data
                self.dict_bck[gfile] = bck
                self.dict_label[gfile] = label
                self.dict_bck_filtered[gfile] = bck_filtered
                self.dict_searched_pattern[gfile] = searched_pattern
            self.label_list.append(self.dict_label[gfile])
            if len(self.dict_searched_pattern[gfile]) != 0:
                self.bck_filtered_list.append(self.dict_bck_filtered[gfile])
                self.searched_pattern_list.append(
                    self.dict_searched_pattern[gfile])

        # Compute distance matrix
        X_train = []
        for gfile in gfile_list:
            X_train.append(self.compute_distmatrix(self.dict_bck[gfile]))

        # Train SVM
        self.clf = SVC(C=self.C, gamma=self.gamma, shrinking=True,
                       class_weight='balanced')
        X_train = preprocessing.scale(X_train, axis=0)
        self.scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = self.scaler.transform(X_train)
        self.clf.fit(X_train, self.label_list)

    def labeling(self, gfile_list):
        y_pred = []
        for gfile in gfile_list:
            yp = self.subject_labeling(gfile)
            y_pred.append(yp)
        return y_pred

    def subject_labeling(self, gfile):
        print('Labeling %s' % gfile)
        # Extract bucket
        if gfile not in self.dict_bck.keys():
            bck_types = ['aims_ss', 'aims_bottom', 'aims_other']
            sbck = []
            graph = aims.read(gfile)
            side = gfile[gfile.rfind('/')+1:gfile.rfind('/')+2]
            trans_tal = aims.GraphManip.talairach(graph)
            vs = graph['voxel_size']
            for vertex in graph.vertices():
                for bck_type in bck_types:
                    if bck_type in vertex:
                        bucket = vertex[bck_type][0]
                        for point in bucket.keys():
                            fpt = [p * v for p, v in zip(point, vs)]
                            trans_pt = list(trans_tal.transform(fpt))
                            if (side == 'R'):
                                trans_pt[0] *= -1
                            trpt_2mm = [int(round(p/2)) for p in trans_pt]
                            sbck.append(trpt_2mm)
        else:
            sbck = self.dict_bck[gfile]
        sbck = np.array(sbck)

        # Compute distance matrix
        X_test = [self.compute_distmatrix(sbck)]

        # Compute classification
        X_test = preprocessing.scale(X_test, axis=0)
        X_test = self.scaler.transform(X_test)
        ypred = self.clf.predict(X_test)

        return ypred

    def find_hyperparameters(self, gfile_list, param_outfile):
        gfile_list = np.asarray(gfile_list)
        best_bacc = 0
        best_C, best_gamma, best_trans = self.C, self.gamma, [0]
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        for C, gamma, trans in itertools.product(
                self.C_range, self.gamma_range, self.trans_range):
            print('------ TEST PARAMETERS ------')
            print('C: %r, gamma: %r, trans: %s' %
                  (C, gamma, str(trans)))
            print()
            self.C = C
            self.gamma = gamma
            self.transrot_init = []
            for x in trans:
                for y in trans:
                    for z in trans:
                        self.transrot_init.append([[1, 0, 0, x],
                                                   [0, 1, 0, y],
                                                   [0, 0, 1, z],
                                                   [0, 0, 0, 1]])
            y_true, y_pred = [], []
            y = [self.dict_label[gfile] for gfile in gfile_list]
            y = np.asarray(y)
            for train, test in skf.split(gfile_list, y):
                print('--- LEARNING (%i samples)' % len(train))
                self.learning(gfile_list[train])
                print()

                print('--- LABELING TEST SET (%i samples)' % len(test))
                y_pred_test = self.labeling(gfile_list[test])
                y_true.extend(y[test])
                y_pred.extend(y_pred_test)
                print()

            bacc = balanced_accuracy(y_true, y_pred, [0, 1])
            if bacc > best_bacc:
                best_bacc = bacc
                best_C = C
                best_gamma = gamma
                best_trans = trans
            print('--- RESULT')
            print('%0.2f for C=%r, gamma=%r, tr=%s' %
                  (bacc, C, gamma, str(trans)))
            print()

        print()
        print('Best parameters set found on development set:')
        print('C=%r, gamma=%r, tr=%s' % (best_C, best_gamma, str(best_trans)))
        print()

        self.C = best_C
        self.gamma = best_gamma
        self.transrot_init = []
        for x in best_trans:
            for y in best_trans:
                for z in best_trans:
                    self.transrot_init.append([[1, 0, 0, x],
                                               [0, 1, 0, y],
                                               [0, 0, 1, z],
                                               [0, 0, 0, 1]])
        param = {'C': best_C,
                 'gamma': best_gamma,
                 'trans': best_trans,
                 'names_filter': self.names_filter,
                 'best_bacc': best_bacc,
                 'pattern': self.pattern,
                 'names_filter': self.names_filter}
        with open(param_outfile, 'w') as f:
            json.dump(param, f)

    def compute_distmatrix(self, sbck):
        X = []
        for bck_filtered, searched_pattern in zip(self.bck_filtered_list,
                                                  self.searched_pattern_list):
            pc1 = pcl.PointCloud(np.asarray(sbck, dtype=np.float32))
            # Try different initialization and select the best score
            dist_min = 100.
            for trans in self.transrot_init:
                pc2 = pcl.PointCloud(np.asarray(apply_trans(
                    trans, bck_filtered), dtype=np.float32))
                bool, transrot, trans_pc2, d = reg.icp(pc2, pc1)
                if d < dist_min:
                    dist_min = d
                    transrot_min = np.dot(transrot, trans)
            trans_searched_pattern = apply_trans(
                transrot_min, searched_pattern)
            X.append(distance_data_to_model(trans_searched_pattern, sbck))
        return X

def apply_trans(transrot, data):
    data = np.asarray(data)
    data_tmp = np.vstack((data.T, np.ones(data.shape[0])))
    new_data_tmp = np.dot(transrot, data_tmp)
    new_data = new_data_tmp[:3].T
    return new_data


def distance_data_to_model(data, model):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(model)
    distances, indices = nbrs.kneighbors(data)
    dist = (distances**2).sum()/len(distances)
    if (dist < 0.000001):
        dist = 0
    return dist
