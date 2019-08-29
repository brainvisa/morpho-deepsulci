# -*- coding: utf-8 -*-
from __future__ import print_function
from ..analyse.stats import balanced_accuracy
from soma import aims
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import time
import copy
import json
import os


class SnipePatternClassification:
    def __init__(self, pattern=None, names_filter=None,
                 n_opal=10, patch_sizes=[6], num_cpu=1):
        self.pattern = pattern
        self.names_filter = names_filter
        self.n_opal = n_opal
        self.patch_sizes = patch_sizes
        self.bck_list = []
        self.label_list = []
        self.gfile_list = []
        self.distmap_list = []

        self.n_opal_range = [5, 10, 15, 20, 25, 30]
        self.patch_sizes_range = [[4], [6], [8],
                                  [4, 6], [6, 8], [4, 8], [4, 6, 8]]
        self.num_cpu = num_cpu

    def learning(self, gfile_list):
        # Extract buckets and labels from the graphs
        bck_types = ['aims_ss', 'aims_bottom', 'aims_other']
        bb = np.array([[100, -100], [100, -100], [100, -100]])
        dict_label = {}
        dict_bck = {}
        label = np.NaN if self.pattern is None else 0
        bck = []
        for gfile in gfile_list:
            if gfile not in self.gfile_list:
                graph = aims.read(gfile)
                side = gfile[gfile.rfind('/')+1:gfile.rfind('/')+2]
                trans_tal = aims.GraphManip.talairach(graph)
                vs = graph['voxel_size']
                label = 0
                bck = []
                bck_filtered = []
                for vertex in graph.vertices():
                    if 'name' in vertex:
                        name = vertex['name']
                        if name.startswith(self.pattern):
                            label = 1
                        filter = sum([1 if name.startswith(n) else 0 for n in self.names_filter])
                        for bck_type in bck_types:
                            if bck_type in vertex:
                                bucket = vertex[bck_type][0]
                                for point in bucket.keys():
                                    fpt = [p * v for p, v in zip(point, vs)]
                                    trans_pt = trans_tal.transform(fpt)
                                    if (side == 'R'):
                                        trans_pt[0] *= -1
                                    trpt_2mm = [int(round(p/2)) for p in list(trans_pt)]
                                    bck.append(trpt_2mm)
                                    if filter:
                                        bck_filtered.append(trpt_2mm)
        
                # compute distmap
                vol_size = 
                vol = aims.Volume_S16(vol_size[0], vol_size[1], vol_size[2])
                vol.fill(0)
                for p in bck:
                    vol[p[0], p[1], p[2]] = 1
                distmap = fm.doit(vol, [0], [1])
                adistmap = np.asarray(distmap)
                adistmap[adistmap > pow(10, 10)] = 0

                # save data
                self.gfile_list.append(gfile)
                self.bck_list.append(bck) ## enlever les doublons ??
                self.label_list.append(label)
                self.distmap_list.append(distmap)
        
    self.y_train
        class_sample_count = np.array(
                [len(np.where(self.y_train == t)[0]) for t in np.unique(self.y_train)])
            w = [max(1., class_sample_count[1]/float(class_sample_count[0])),
                 max(1., class_sample_count[0]/float(class_sample_count[1]))]
            self.proba_list = [w[yi] for yi in self.y_train]
                        bck_list, names_list, bckmap_list = [], [], []
            voronoi_list, vol_list = [], []
            for d, subj in self.train_data:
                bck_file = os.path.join(d, 'buckets_2mm/'+subj+'.npy')
                names_file = os.path.join(d, 'names_2mm/'+subj+'.npy')
                bckmap_file = os.path.join(d, 'bucketmap_2mm/'+subj+'.bck')
                bckmap_list.append(aims.read(bckmap_file))
                bck_list.append(np.load(bck_file))
                names = np.load(names_file)
                names_list.append(np.asarray([n if n.startswith(self.sulcus)
                                              else 'unknown'
                                              for n in names]))
                voronoi_file = os.path.join(d, 'voronoi_2mm/'+subj+'.nii')
                vol_file = os.path.join(d, self.feature+'/'+subj+'.nii')
                voronoi_list.append(aims.read(voronoi_file))
                vol_list.append(aims.read(vol_file))
                
                
                        vol = aims.Volume_S16(vol_list[0].getSizeX(),
                              vol_list[0].getSizeY(),
                              vol_list[0].getSizeZ())
        for bck, names in zip(bck_list, names_list):
            sulcus_bck = bck[names != 'unknown']
            for p in sulcus_bck:
                vol[p[0], p[1], p[2]] = 1
        dilation = aimsalgo.MorphoGreyLevel_S16()
        mask = dilation.doDilation(vol, self.mask)


    # find boundingbox of the whole brain
    bb_brain = np.array([[100., -100.], [100., -100.], [100., -100.]])
    for idx in subject_list:
        bck = np.load(os.path.join(db_path+'/buckets', idx+'.npy'))
        for i in range(3):
            bb_brain[i, 0] = min(bb_brain[i, 0], int(min(bck[:, i]))-1)
            bb_brain[i, 1] = max(bb_brain[i, 1], int(max(bck[:, i]))+1)

    if translation is None:
        print('compute translation...')
        tr = [-int(bb_brain[i, 0]/2) + 11 for i in range(3)]
    else:
        tr = translation


    def labeling(self, gfile_list):
        gfile_list = np.asarray(gfile_list)
        n_splits = min(self.num_cpu, len(gfile_list))
        if n_splits > 1:
            children = []
            kf = KFold(n_splits=n_splits)
            i = 1
            for train_index, test_index in kf.split(gfile_list):
                if i == n_splits:
                    break
                child = os.fork()
                if child:
                    children.append(child)
                else:
                    for d, s in gfile_list[test_index]:
                        self.subject_labelisation(
                            d, s, save_all, opal_precomputed, outpath)
                    print('CHILD END')
                    os._exit(0)
                i += 1
            for d, s in gfile_list[test_index]:
                self.subject_labelisation(
                    d, s, save_all, opal_precomputed, outpath)
            for child in children:
                while True:
                    try:
                        sp = os.waitpid(child, 0)
                        break
                    except OSError as err:
                        print("OS error: {0}".format(err))
        else:
            for d, s in gfile_list:
                self.subject_labelisation(
                    d, s, save_all, opal_precomputed, outpath)

        # result file
        r = pd.DataFrame(index=[str(s) for d, s in gfile_list],
                         columns=['y_pred', 'y_score'])
        for d, subject in gfile_list:
            op = '/neurospin/tmp/lborne' if outpath is None else outpath
            outdir = os.path.join(op, subject)
            result_path = os.path.join(outdir, 'result.csv')
            df_result = pd.read_csv(result_path, index_col=0)
            y_score = np.nanmean(df_result['grade']) ## fix sum !!
            y_pred = 1 if y_score > 0 else 0
            r.set_value(subject, 'y_score', y_score)
            r.set_value(subject, 'y_pred', y_pred)

        return y_pred, y_score

    def subject_labelisation(self, gfile):
        sbck = 
        sdistmap = 
        sbck = np.asarray(apply_imask(sbck, mask))

        grading_list = []
        for ps in self.patch_sizes:
            opm = OptimizedPatchMatch(
                patch_size=[ps, ps, ps], search_size=self.search_size,
                segmentation=False, k=self.n_opal, j=4)
            list_dfann = opm.run(
                distmap=sdistmap, bck=sbck,
                distmap_list=self.distmap_list, bck_list=self.bck_list,
                proba_list=self.proba_list)
            grading_list.append(grading(sbck, list_dfann, grade_list))

        grade = np.nanmean(np.mean(grading_list, axis=0))
        ypred = 1 if grade > 0 else 0
        return ypred, grade

    def find_hyperparameters(self, gfile_list, param_outfile):
        best_bacc = 0
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        for n_opal, patch_sizes in itertools.product(
                self.n_opal_range, self.patch_sizes_range):
            self.n_opal = n_opal
            self.patch_sizes = patch_sizes
            ps = ''.join([str(i) for i in patch_sizes])
            y_true, y_pred = [], []
            for train, test in skf.split(gfile_list, y): ## y
                self.learning(gfile_list[train])
                y_pred_test, _ = self.labelisation(gfile_list[test])
                y_true.extend(y[test])
                y_pred.extend(y_pred_test)

            bacc = bacc_score(y_true, y_pred)
            if bacc > best_bacc:
                best_bacc = bacc
                self.best_n_opal = n_opal
                self.best_patch_sizes = patch_sizes
            print('%0.2f for n_opal=%r, patch_sizes=%s' % (bacc, n_opal, ps))

        print()
        print('Best parameters set found on development set:')
        print('n_opal=%r, patch_sizes=%s' % (self.best_n_opal,
                                             str(self.best_patch_sizes)))
        print()

        self.n_opal = self.best_n_opal
        self.patch_sizes = self.best_patch_sizes
        param = {'n_opal': self.best_n_opal,
                 'patch_sizes': self.best_patch_sizes,
                 'best_bacc': best_bacc}
        with open(param_outfile, 'w') as f:
            json.dump(param, f)


def apply_imask(ipoints, mask):
    ipoints, _ = apply_bounding_box(
        ipoints, np.transpose([[0, 0, 0], np.array(mask.getSize())[:3]-1]))
    amask = np.asarray(mask)
    values = amask[:, :, :, 0][ipoints.T[0], ipoints.T[1], ipoints.T[2]]
    return ipoints[values == 1]


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
