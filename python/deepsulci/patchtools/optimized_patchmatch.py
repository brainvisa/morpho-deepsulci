# -*- coding: utf-8 -*-
from __future__ import print_function
from soma import aims
import pandas as pd
import numpy as np
import random
import math
import time


class OptimizedPatchMatch:
    def __init__(self, patch_size, search_size=[3, 3, 3],
                 border=10, segmentation=True, k=5, j=4):
        self.patch_size = patch_size
        self.search_size = search_size
        self.seg = segmentation
        self.k = k
        self.j = j
        self.dim = 3
        self.col_dim = ['point_x', 'point_y', 'point_z']
        self.ann_col_dim = ['ann_point_x', 'ann_point_y', 'ann_point_z']
        self.border = border

    def run(self, distmap, distmap_list, bck_list,
            bckmap_list=[], dnum={}, proba_list=None):

        list_df_ann = []
        for k in range(self.k):
            start_time = time.clock()
            df_ann = self.initialization(distmap, dnum, distmap_list, bck_list,
                                         bckmap_list, proba_list)
            for j in range(self.j):
                df_ann = self.iteration(j, df_ann, distmap, distmap_list,
                                        bckmap_list, dnum)
            list_df_ann.append(df_ann)
            print('OPM (%i/%i) took %i s.' %
                  (k+1, self.k, time.clock()-start_time))
        return list_df_ann

    def initialization(self, distmap, dnum,
                       distmap_list, bck_list, bckmap_list, proba_list):
        bck = np.transpose(np.where(np.asarray(distmap) == 0)[:3])
        df_ann = pd.DataFrame(index=range(len(bck)),
                              columns=['ann_point_x', 'ann_point_y',
                                       'ann_point_z', 'ann_name', 'ann_num',
                                       'dist', 'prop_num'])
        for d in range(self.dim):
            df_ann[self.col_dim[d]] = bck[:, d]
        df_ann['prop_num'] = range(len(bck))

        i = 0
        for ipt, j in zip(bck, range(len(bck))):
            patch = self.vol_compute_patch(distmap, ipt, self.patch_size)
            if proba_list is None:
                rand_num = random.choice(range(len(bck_list)))
            else:
                rand_num = int(round(np.random.choice(
                    range(len(bck_list)), 1,
                    p=np.array(proba_list)/float(np.sum(proba_list)))[0]))
            s = 0
            while True:
                i += 1
                inpoints = self.compute_patch(distmap_list[rand_num], ipt,
                                              np.asarray(self.search_size) + s)
                if len(inpoints) != 0:
                    break
                else:
                    if s < self.border - self.search_size[0]:
                        s += 1
                    else:
                        s = 0
                        rand_num = random.choice(range(len(bck_list)))
            rand_point = random.choice(inpoints)
            rand_patch = self.vol_compute_patch(distmap_list[rand_num],
                                                rand_point, self.patch_size)
            for d, rp in zip(self.ann_col_dim, rand_point):
                df_ann.set_value(j, d, rp)
            df_ann.set_value(j, 'ann_num', rand_num)
            df_ann.set_value(j, 'dist', self.distance(patch, rand_patch))
            if self.seg:
                n = bckmap_list[rand_num][0][rand_point]
                df_ann.set_value(j, 'ann_name', dnum[n])
        return df_ann

    def iteration(self, j, df_ann, distmap, distmap_list, bckmap_list, dnum):

        self.nprop, self.nsearch = 0, 0
        if j % 2 == 0:
            df_ann.sort_values(by=list(self.col_dim),
                               inplace=True, ascending=True)
        else:
            df_ann.sort_values(by=list(self.col_dim),
                               inplace=True, ascending=False)
        self.points_prop, self.ind_prop = [], []
        for idx in df_ann.index:
            point = np.asarray(df_ann.loc[idx, self.col_dim], dtype=int)
            patch = self.vol_compute_patch(distmap, point, self.patch_size)
            # Propagation step
            df_ann = self.propagation(
                         idx, patch, point, df_ann, distmap_list,
                         bckmap_list, dnum)
            # Constrained random search
            df_ann = self.search(idx, patch, df_ann, distmap_list,
                                 bckmap_list, dnum)
        return df_ann

    def propagation(self, idx, patch, point, df_ann,
                    distmap_list, bckmap_list, dnum):
        prop_bb = np.transpose([point - 1, point + 2])
        if len(self.points_prop) != 0:
            npoints, nidx = self.apply_bounding_box(
                self.points_prop, prop_bb)
            if len(npoints) != 0:
                for npoint, ind in zip(npoints,
                                       np.asarray(self.ind_prop)[nidx]):
                    if ind != idx:
                        ann_point = np.asarray(df_ann.loc[ind,
                                                          self.ann_col_dim])
                        ann_num = df_ann.at[ind, 'ann_num']
                        virtual_point = ann_point + (point - npoint)
                        val = distmap_list[ann_num].at(virtual_point)
                        if val != 0:
                            val = int(math.ceil(val)) + 1
                            inpoints = self.compute_patch(
                                distmap_list[ann_num], virtual_point,
                                [val]*self.dim)
                            new_point = random.choice(inpoints)
                        else:
                            new_point = virtual_point

                        new_patch = self.vol_compute_patch(
                            distmap_list[ann_num], new_point,
                            self.patch_size)
                        dist = self.distance(patch, new_patch)
                        if dist < df_ann.at[idx, 'dist']:
                            self.nprop += 1
                            for d, rp in zip(self.ann_col_dim, new_point):
                                df_ann.set_value(idx, d, rp)
                            df_ann.set_value(idx, 'ann_num', ann_num)
                            df_ann.set_value(idx, 'dist', dist)
                            pn = df_ann.at[ind, 'prop_num']
                            df_ann.set_value(idx, 'prop_num', pn)
                            if self.seg:
                                n = bckmap_list[ann_num][0][new_point]
                                df_ann.set_value(idx, 'ann_name', dnum[n])
        self.points_prop.append(point)
        self.ind_prop.append(idx)
        return df_ann

    def search(self, idx, patch, df_ann, distmap_list, bckmap_list, dnum):
        ann_point = np.asarray(df_ann.loc[idx, self.ann_col_dim])
        ann_num = df_ann.at[idx, 'ann_num']
        for s in [10, 5, 2, 1]:
            inpoints = self.compute_patch(distmap_list[ann_num], ann_point,
                                          [s]*self.dim)
            rand_point = random.choice(inpoints)
            # test if the patch is better
            new_patch = self.vol_compute_patch(
                distmap_list[ann_num], rand_point, self.patch_size)
            dist = self.distance(patch, new_patch)
            if dist < df_ann.at[idx, 'dist']:
                self.nsearch += 1
                for d, rp in zip(self.ann_col_dim, rand_point):
                    df_ann.set_value(idx, d, rp)
                df_ann.set_value(idx, 'dist', dist)
                if self.seg:
                    n = bckmap_list[ann_num][0][rand_point]
                    df_ann.set_value(idx, 'ann_name', dnum[n])
        return df_ann

    def compute_patch(self, vol, point, patch_size):
        invol = self.vol_compute_patch(vol, point, patch_size)
        corner_point = np.asarray(point)-np.asarray(patch_size)
        inpoints = np.transpose(np.where(np.asarray(invol) == 0))[:, :self.dim]
        return inpoints + corner_point

    def vol_compute_patch(self, vol, ipoint, patch_size):
        vol_size = np.asarray(patch_size)*2+1
        corner_point = np.asarray(ipoint)-np.asarray(patch_size)
        return aims.VolumeView(vol, corner_point, vol_size)

    def apply_bounding_box(self, points, bb):
        bb = np.asarray(bb)
        points = np.asarray(points)
        inidx = np.all(np.logical_and(bb[:, 0] <= points, points < bb[:, 1]),
                       axis=1)
        inbox = points[inidx]
        return inbox, np.asarray(range(len(points)))[inidx]

    def distance(self, distmap0, distmap1):
        d0 = np.mean(distmap1[np.asarray(distmap0) == 0]**2)
        d1 = np.mean(distmap0[np.asarray(distmap1) == 0]**2)
        return np.mean([d0, d1])
