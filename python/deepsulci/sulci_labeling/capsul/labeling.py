from __future__ import print_function
from ..method.cutting import cutting
from ..method.unet import UnetSulciLabeling
from soma.aimsalgo.sulci import graph_pointcloud
from soma import aims
from capsul.api import Process
from collections import Counter
import traits.api as traits
import pandas as pd
import numpy as np
import time
import json


class SulciDeepLabeling(Process):
    def __init__(self):
        super(SulciDeepLabeling, self).__init__()
        self.add_trait('graph', traits.File(output=False))
        self.add_trait('roots', traits.File(output=False))
        self.add_trait('param_file', traits.File(output=False))
        self.add_trait('model_file', traits.File(output=False))

        self.add_trait('labeled_graph', traits.File(output=True))

    def _run_process(self):
        start_time = time.time()
        with open(self.param_file) as f:
            param = json.load(f)
        self.sulci_side_list = param['sulci_side_list']
        method = UnetSulciLabeling(
            self.sulci_side_list, num_filter=64, batch_size=1, cuda=None)
        method.load(self.model_file)

        dict_sulci = {self.sulci_side_list[i]: i for i in range(len(
            self.sulci_side_list))}
        dict_num = {v: k for k, v in dict_sulci.items()}

        # voxel labeling
        y_pred, y_scores, vert, bck, bck2 = method.labeling(
            self.graph, rypred=True, ryscores=True, rvert=True,
            rbck=True, rbck2=True)

        # cutting
        # TODO. verifier que ca marche toujours avec les doublons !!
        np.save('/tmp/test_new/vert.npy', vert)
        np.save('/tmp/test_new/y_scores.npy', y_scores)
        np.save('/tmp/test_new/bck2.npy', bck2)
        print('threshold', param['cutting_threshold'])
        y_pred_cut = cutting(
            y_scores, vert, bck2, threshold=param['cutting_threshold'])

        # conversion to Talairach
        for i in set(vert):
            c = Counter(y_pred_cut[vert == i])
            if len(c) > 1:
                if c.most_common()[1][1] < 20:
                    predicted_label = c.most_common()[0][0]
                    y_pred_cut[vert == i] = predicted_label

        # graph conversion
        graph = aims.read(self.graph)
        data = pd.DataFrame()
        bck = np.asarray(bck)
        data['point_x'] = bck[:, 0]
        data['point_y'] = bck[:, 1]
        data['point_z'] = bck[:, 2]
        data['before_cutting'] = [dict_num[y] for y in y_pred]
        data['after_cutting'] = [dict_num[y] for y in y_pred_cut]
        # TODO. remove saving
        data.to_csv('/tmp/test_new/result.csv')
        roots = aims.read(self.roots)
        graph, summary = graph_pointcloud.build_split_graph(
            graph, data, roots)

        # save graph
        aims.write(graph, self.labeled_graph)
        print('Labeling took %i sec.' % (time.time() - start_time))
