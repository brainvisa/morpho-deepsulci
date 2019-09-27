'''
UNET Labeling Module
'''

from __future__ import print_function
from ...deeptools.dataset import extract_data
from ..method.cutting import cutting
from ..method.unet import UnetSulciLabeling
from soma.aimsalgo.sulci import graph_pointcloud
from soma import aims
from soma import aimsalgo
from capsul.api import Process
from collections import Counter
import traits.api as traits
import pandas as pd
import numpy as np
import time
import json


class SulciDeepLabeling(Process):
    '''
    Process to label a new graph using a 3D U-Net convolutional neural network.
    '''

    def __init__(self):
        super(SulciDeepLabeling, self).__init__()
        self.add_trait('graph', traits.File(
            output=False, desc='input graph to segment'))
        self.add_trait('roots', traits.File(
            output=False, desc='root file corresponding to the input graph'))
        self.add_trait('model_file', traits.File(
            output=False, desc='file (.mdsm) storing neural network'
                               ' parameters'))
        self.add_trait('param_file', traits.File(
            output=False, desc='file (.json) storing the hyperparameters'
                               ' (cutting threshold)'))
        self.add_trait('rebuild_attributes', traits.Bool(True, output=False))
        self.add_trait('skeleton', traits.File(
            output=False,
            desc='skeleton file corresponding to the input graph'))
        self.add_trait('allow_multithreading', traits.Bool(True, output=False))

        self.add_trait('labeled_graph', traits.File(
            output=True, desc='output labeled graph'))

    def _run_process(self):
        start_time = time.time()
        with open(self.param_file) as f:
            param = json.load(f)
        self.sulci_side_list = param['sulci_side_list']
        method = UnetSulciLabeling(
            self.sulci_side_list, num_filter=64, batch_size=1, cuda=-1)
        method.load(self.model_file)

        dict_sulci = {self.sulci_side_list[i]: i for i in range(len(
            self.sulci_side_list))}
        dict_num = {v: k for k, v in dict_sulci.items()}

        # voxel labeling
        graph = aims.read(self.graph)
        data = extract_data(graph)
        data = {k: np.asarray(v) for k, v in data.iteritems()}

        _, y_pred, y_scores = method.labeling(
            self.graph, data['bck2'], ['unknown']*len(data['bck2']))

        # cutting
        print('threshold', param['cutting_threshold'])
        y_pred_cut = cutting(
            y_scores, data['vert'], data['bck2'],
            threshold=param['cutting_threshold'])

        # conversion to Talairach
        for i in set(data['vert']):
            c = Counter(y_pred_cut[data['vert'] == i])
            if len(c) > 1:
                if c.most_common()[1][1] < 20:
                    predicted_label = c.most_common()[0][0]
                    y_pred_cut[data['vert'] == i] = predicted_label

        # graph conversion
        graph = aims.read(self.graph)
        result = pd.DataFrame()
        bck = data['bck']
        result['point_x'] = bck[:, 0]
        result['point_y'] = bck[:, 1]
        result['point_z'] = bck[:, 2]
        result['before_cutting'] = [dict_num[y] for y in y_pred]
        result['after_cutting'] = [dict_num[y] for y in y_pred_cut]
        roots = aims.read(self.roots)
        graph, summary = graph_pointcloud.build_split_graph(
            graph, result, roots)

        print('summary:', summary)
        if self.rebuild_attributes and summary['cuts'] != 0:
            skel = aims.read(self.skeleton, 1)
            inside = 0
            outside = 11
            fat = aims.FoldGraphAttributes(skel, graph, None, inside, outside)
            if hasattr(fat, 'setMaxThreads'):
                if self.allow_multithreading:
                    threads = 0
                else:
                    threads = 1
                fat.setMaxThreads(threads)
            smoothType = aimsalgo.Mesher.LOWPASS
            fat.mesher().setSmoothing(smoothType, 50, 0.4)
            fat.mesher().setDecimation(100., 2., 3., 180.0)
            fat.doAll()

        graph['label_property'] = 'label'
        # save graph
        aims.write(graph, self.labeled_graph)
        print('Labeling took %i sec.' % (time.time() - start_time))
