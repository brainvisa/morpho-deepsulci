# -*- coding: utf-8 -*-
from __future__ import print_function
from torch.utils.data import Dataset
from soma import aims
import numpy as np
import torch
import random
import math
import sigraph


class SulciDataset(Dataset):
    def __init__(self, gfile_list, dict_sulci,
                 train=True, translation_file=None,
                 dict_bck2={}, dict_names={}):
        self.gfile_list = gfile_list
        self.dict_sulci = dict_sulci
        if 'background' not in self.dict_sulci.keys():
            self.dict_sulci['background'] = -1
        self.train = train
        self.rot_angle = math.pi/16
        self.translation_file = translation_file
        self.dict_bck2 = dict_bck2
        self.dict_names = dict_names

    def transform(self, bck):
        # rotation
        if self.rot_angle is not None:
            center = (np.max(bck, axis=0) - np.min(bck, axis=0))/2
            transrot = random_rotation(center, self.rot_angle)
            bck = rotation_bck(bck, transrot)
        # translation
        tr = np.min(bck, axis=0)
        bck -= tr

        return bck

    def __getitem__(self, index):
        gfile = self.gfile_list[index]
        if gfile in self.dict_bck2.keys():
            bck2 = np.asarray(self.dict_bck2[gfile])
            names = np.asarray(self.dict_names[gfile])
        else:
            graph = aims.read(gfile)
            if self.translation_file is not None:
                flt = sigraph.FoldLabelsTranslator()
                flt.readLabels(self.translation_file)
                flt.translate(graph)

            # extract bucket/names
            data = extract_data(graph)
            bck2 = np.asarray(data['bck2'])
            names = np.asarray(data['names'])
            self.dict_bck2[gfile] = bck2
            self.dict_names[gfile] = names

        tr = np.min(bck2, axis=0)
        bck2 = bck2 - tr

        # data augmentation
        if self.train:
            bck2 = self.transform(bck2)

        # input/label
        bck2 = np.array(bck2, dtype=int)
        bck_T = bck2.transpose()
        img_size = np.max(bck2, axis=0) + 1
        input = torch.zeros(
            1, img_size[0], img_size[1], img_size[2], dtype=torch.float)
        input[0][bck_T[0], bck_T[1], bck_T[2]] = 1

        labels = torch.zeros(
            img_size[0], img_size[1], img_size[2], dtype=torch.long)
        labels += self.dict_sulci['background']
        labels[bck_T[0], bck_T[1], bck_T[2]] = torch.tensor(
            [self.dict_sulci[n] for n in names], dtype=torch.long)

        return input, labels

    def __len__(self):
        return len(self.gfile_list)


class PatternDataset(Dataset):
    def __init__(self, gfile_list, pattern, bb, train=True,
                 dict_bck={}, dict_label={}, labels=None):
        self.gfile_list = gfile_list
        self.labels = labels
        self.pattern = pattern
        self.bb = np.array(bb)
        self.size = self.bb[:, 1] - self.bb[:, 0] + 1
        self.tr = self.bb[:, 0]
        self.rot_angle = math.pi/40
        self.tr_sigma = 2
        self.train = train
        self.dict_bck = dict_bck
        self.dict_label = dict_label

    def transform(self, bck):
        # rotation
        if self.rot_angle is not None:
            center = (np.max(bck, axis=0) - np.min(bck, axis=0))/2
            transrot = random_rotation(center, self.rot_angle)
            bck = rotation_bck(bck, transrot)
        # translation
        tr = [int(round(np.random.normal(0, self.tr_sigma))),
              int(round(np.random.normal(0, self.tr_sigma))),
              int(round(np.random.normal(0, self.tr_sigma)))]
        bck += tr

        return bck

    def __getitem__(self, index):
        gfile = self.gfile_list[index]

        # Bucket extraction
        if gfile in self.dict_bck.keys():
            bck = self.dict_bck[gfile]
            label = self.dict_label[gfile]
        else:
            side = gfile[gfile.rfind('/')+1:gfile.rfind('/')+2]
            flip = True if side == 'R' else False
            graph = aims.read(gfile)
            data = extract_data(graph, flip=flip)
            bck = data['bck2']
            if self.labels is not None:
                label = self.labels[index]
            elif self.pattern is None:
                label = np.NaN
            else:
                label = 0
                for n in data['names']:
                    if n.startswith(self.pattern):
                        label = 1
                        break

            self.dict_bck[gfile] = bck
            self.dict_label[gfile] = label

        # Data augmentation
        if self.train:
            bck = self.transform(bck)

        # Pytorch output
        bck, _ = apply_bounding_box(bck, self.bb)
        bck -= self.tr
        values = np.ones(len(bck))
        bck_T = bck.transpose()
        input = torch.zeros(
            1, self.size[0], self.size[1], self.size[2], dtype=torch.float)
        input[0][bck_T[0], bck_T[1], bck_T[2]] = torch.tensor(
            values, dtype=torch.float)

        return input, label

    def __len__(self):
        return len(self.gfile_list)


def extract_data(graph, flip=False):
    trans_tal = aims.GraphManip.talairach(graph)
    vs = graph['voxel_size']
    bck_types = ['aims_ss', 'aims_bottom', 'aims_other']
    data = {'bck': [], 'nbck': [], 'bck2': [], 'vert': [], 'names': []}
    for vertex in graph.vertices():
        if 'name' in vertex:
            name = vertex['name']
        else:
            name = 'unknown'
        for bck_type in bck_types:
            if bck_type in vertex:
                bucket = vertex[bck_type][0]
                for point in bucket.keys():
                    if flip:
                        point[0] *= -1
                    data['nbck'].append(point)
                    p0 = [p * v for p, v in zip(point, vs)]
                    p1 = trans_tal.transform(p0)
                    data['bck'].append(list(p1))
                    p2 = [int(round(p1[i]/2)) for i in range(3)]
                    data['bck2'].append(p2)
                    data['names'].append(name)
                    data['vert'].append(vertex['index'])
    return data


def apply_bounding_box(points, bb):
    bb = np.asarray(bb)
    points = np.asarray(points)
    inidx = np.all(np.logical_and(bb[:, 0] <= points, points <= bb[:, 1]),
                   axis=1)
    inbox = points[inidx]
    return inbox, np.asarray(range(len(points)))[inidx]


def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data


def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
    ...                                               direc, point)))
    True

    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[0.0, -direction[2], direction[1]],
                   [direction[2], 0.0, -direction[0]],
                   [-direction[1], direction[0], 0.0]])
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def random_rotation(center, rot_angle):
    th = random.uniform(0, 2*math.pi)
    z = random.uniform(-1, 1)
    direction = [np.sqrt(1-z**2)*np.cos(th), np.sqrt(1-z**2)*np.sin(th), z]
    transrot = rotation_matrix(
        np.random.normal(0, rot_angle),
        direction,
        np.array(center))
    return transrot


def rotation_bck(bck, transrot):
    bck_4d = np.vstack([np.transpose(bck), np.ones(len(bck))])
    bck_4d_trans = np.dot(transrot, bck_4d)
    bck = np.transpose(bck_4d_trans[:-1])
    bck = np.array(bck, dtype=int)
    return bck
