# -*- coding: utf-8 -*-
from __future__ import print_function
from ..analyse.stats import balanced_accuracy
from ...patchtools.optimized_patchmatch import OptimizedPatchMatch
from soma import aims, aimsalgo
from sklearn.model_selection import StratifiedKFold

import numpy as np
import json
import itertools


class SnipePatternClassification:
    def __init__(self, pattern=None, names_filter=None,
                 n_opal=10, patch_sizes=[6], num_cpu=1,
                 dict_bck=None, dict_bck_filtered=None, dict_label=None):
        self.pattern = pattern
        self.nfilter = names_filter
        self.n_opal = n_opal
        self.patch_sizes = patch_sizes
        if dict_bck is None:
            self.dict_bck = {}
        else:
            self.dict_bck = dict_bck
        self.bck_list = []
        if dict_bck_filtered is None:
            self.dict_bck_filtered = {}
        else:
            self.dict_bck_filtered = dict_bck_filtered
        if dict_label is None:
            self.dict_label = {}
        else:
            self.dict_label = dict_label
        self.label_list = []
        self.distmap_list = []

#        self.n_opal_range = [5, 10, 15, 20, 25, 30]
        self.n_opal_range = [2, 4]
#        self.patch_sizes_range = [[4], [6], [8],
#                                  [4, 6], [6, 8], [4, 8], [4, 6, 8]]
        self.patch_sizes_range = [[4], [4, 6]]
        self.num_cpu = num_cpu

    def learning(self, gfile_list):
        self.bck_list, self.label_list, self.distmap_list = [], [], []
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
                bck = []
                bck_filtered = []
                for vertex in graph.vertices():
                    if 'name' in vertex:
                        n = vertex['name']
                        if n.startswith(self.pattern):
                            label = 1
                        nf = sum([1 for m in self.nfilter if n.startswith(m)])
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

                # save data
                self.dict_bck[gfile] = bck
                self.dict_label[gfile] = label
                self.dict_bck_filtered[gfile] = bck_filtered
            self.bck_list.append(self.dict_bck[gfile])
            self.label_list.append(self.dict_label[gfile])

        # Compute volume size + mask
        bb = np.array([[100., -100.], [100., -100.], [100., -100.]])
        for gfile in gfile_list:
            abck = np.asarray(self.dict_bck[gfile])
            for i in range(3):
                bb[i, 0] = min(bb[i, 0], int(min(abck[:, i]))-1)
                bb[i, 1] = max(bb[i, 1], int(max(abck[:, i]))+1)
        self.translation = [-int(bb[i, 0]) + 11 for i in range(3)]
        self.vol_size = [int((bb[i, 1] - bb[i, 0]))+22 for i in range(3)]

        # Compute distmap volumes + mask
        fm = aims.FastMarching()
        vol_filtered = aims.Volume_S16(
            self.vol_size[0], self.vol_size[1], self.vol_size[2])
        for gfile in gfile_list:
            bck = np.asarray(self.dict_bck[gfile]) + self.translation
            bck_filtered = np.asarray(self.dict_bck_filtered[gfile])
            if len(bck_filtered) != 0:
                bck_filtered += self.translation
            # compute distmap
            vol = aims.Volume_S16(
                self.vol_size[0], self.vol_size[1], self.vol_size[2])
            vol.fill(0)
            for p in bck:
                vol[p[0], p[1], p[2]] = 1
            distmap = fm.doit(vol, [0], [1])
            adistmap = np.asarray(distmap)
            adistmap[adistmap > pow(10, 10)] = 0
            self.distmap_list.append(distmap)
            # compute mask
            for p in bck_filtered:
                vol_filtered[p[0], p[1], p[2]] = 1
        dilation = aimsalgo.MorphoGreyLevel_S16()
        self.mask = dilation.doDilation(vol_filtered, 5)

        # Compute proba_list
        y_train = self.label_list
        class_sample_count = np.array(
            [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        w = [max(1., class_sample_count[1]/float(class_sample_count[0])),
             max(1., class_sample_count[0]/float(class_sample_count[1]))]
        self.proba_list = [w[yi] for yi in y_train]

    def labeling(self, gfile_list):
        y_pred, y_score = [], []
        for gfile in gfile_list:
            yp, ys = self.subject_labeling(gfile)
            y_pred.append(yp)
            y_score.append(ys)

        return y_pred, y_score

    def subject_labeling(self, gfile):
        print('Labeling %s' % gfile)
        # Extract bucket
        fm = aims.FastMarching()
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
        sbck += self.translation
        sbck = np.asarray(apply_imask(sbck, self.mask))

        # Extract distmap
        fm = aims.FastMarching()
        vol = aims.Volume_S16(
            self.vol_size[0], self.vol_size[1], self.vol_size[2])
        vol.fill(0)
        for p in sbck:
            vol[p[0], p[1], p[2]] = 1
        sdistmap = fm.doit(vol, [0], [1])
        adistmap = np.asarray(sdistmap)
        adistmap[adistmap > pow(10, 10)] = 0

        # Compute classification
        grading_list = []
        for ps in self.patch_sizes:
            print('** PATCH SIZE %i' % ps)
            opm = OptimizedPatchMatch(
                patch_size=[ps, ps, ps], segmentation=False, k=self.n_opal)
            list_dfann = opm.run(
                distmap=sdistmap, bck=sbck,
                distmap_list=self.distmap_list, bck_list=self.bck_list,
                proba_list=self.proba_list)
            grading_list.append(grading(sbck, list_dfann, self.label_list))

        grade = np.nanmean(np.mean(grading_list, axis=0))
        ypred = 1 if grade > 0 else 0
        return ypred, grade

    def find_hyperparameters(self, gfile_list, param_outfile):
        gfile_list = np.asarray(gfile_list)
        best_bacc = 0
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        for n_opal, patch_sizes in itertools.product(
                self.n_opal_range, self.patch_sizes_range):
            print('------ TEST PARAMETERS ------')
            print('number of OPM iterations: %i, patch sizes: %s' %
                  (n_opal, str(patch_sizes)))
            print()
            self.n_opal = n_opal
            self.patch_sizes = patch_sizes
            y_true, y_pred = [], []
            y = [self.dict_label[gfile] for gfile in gfile_list]
            y = np.asarray(y)
            for train, test in skf.split(gfile_list, y):
                print('--- LEARNING (%i samples)' % len(train))
                self.learning(gfile_list[train])
                print()

                print('--- LABELING TEST SET (%i samples)' % len(test))
                y_pred_test, _ = self.labeling(gfile_list[test])
                y_true.extend(y[test])
                y_pred.extend(y_pred_test)
                print()

            bacc = balanced_accuracy(y_true, y_pred, [0, 1])
            if bacc > best_bacc:
                best_bacc = bacc
                self.best_n_opal = n_opal
                self.best_patch_sizes = patch_sizes
            print('--- RESULT')
            print('%0.2f for n_opal=%r, patch_sizes=%s' %
                  (bacc, n_opal, str(patch_sizes)))
            print()

        print()
        print('Best parameters set found on development set:')
        print('n_opal=%r, patch_sizes=%s' % (self.best_n_opal,
                                             str(self.best_patch_sizes)))
        print()

        self.n_opal = self.best_n_opal
        self.patch_sizes = self.best_patch_sizes
        param = {'n_opal': self.best_n_opal,
                 'patch_sizes': self.best_patch_sizes,
                 'pattern': self.pattern,
                 'names_filter': self.names_filter,
                 'best_bacc': best_bacc}
        with open(param_outfile, 'w') as f:
            json.dump(param, f)


def apply_imask(ipoints, mask):
    ipoints, _ = apply_bounding_box(
        ipoints, np.transpose([[0, 0, 0], np.array(mask.getSize())[:3]-1]))
    amask = np.asarray(mask)
    values = amask[:, :, :, 0][ipoints.T[0], ipoints.T[1], ipoints.T[2]]
    return ipoints[values == 1]


def apply_bounding_box(points, bb):
    bb = np.asarray(bb)
    points = np.asarray(points)
    inidx = np.all(np.logical_and(bb[:, 0] <= points, points < bb[:, 1]),
                   axis=1)
    inbox = points[inidx]
    return inbox, np.asarray(range(len(points)))[inidx]


def grading(points, list_dfann, grade_list):
    idxs = range(len(points))
    grad_list = np.zeros(len(points))

    for pt, idx in zip(points, idxs):
        distances, names = [], []
        for k in range(len(list_dfann)):
            df_ann = list_dfann[k]
            distances.append(df_ann.loc[idx, 'dist'])
            ann_num = int(round(df_ann.loc[idx, 'ann_num']))
            names.append(grade_list[ann_num])

        # prediction
        v = 0
        weights = []
        h = min(distances)+0.1
        for d, n in zip(distances, names):
            w = np.exp(-pow(d, 2)/pow(h, 2))
            weights.append(w)
            if n == 1:
                v += w
            else:
                v -= w
        grad_list[idx] = v / np.sum(weights)
    return grad_list
