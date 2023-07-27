'''
UNET Labeling Module
'''

from capsul.api import Process
from soma.controller import File, field


class SulciDeepLabeling(Process):
    '''
    Process to label a new graph using a 3D U-Net convolutional neural network.

    The process can work using a GPU or on CPU. It requires a fair amount of
    RAM memory (about 4-5 GB). If not enough memory can be allocated, the
    process will abort with an error (thus will not hang the whole machine).
    '''

    graph: File = field(type_=File, doc='input graph to segment')
    roots: File = field(type_=File,
                        doc='root file corresponding to the input graph')
    model_file: File = field(type_=File,
                             doc='file (.mdsm) storing neural network'
                                 ' parameters')
    param_file: File = field(type_=File,
                             doc='file (.json) storing the hyperparameters'
                                 ' (cutting threshold)')
    rebuild_attributes: bool = False
    skeleton: File = field(
        type_=File,
        doc='skeleton file corresponding to the input graph')
    allow_multithreading: bool = True

    labeled_graph: File = field(type_=File, write=True,
                                doc='output labeled graph')
    cuda: int = field(default=-1,
                      doc='device on which to run the training'
                          '(-1 for cpu, i>=0 for the i-th gpu)')
    fix_random_seed: bool = field(default=False,
                                  doc='Use same random sequence')

    def execution(self, context=None):
        from ...deeptools.dataset import extract_data
        from ..method.cutting import cutting
        from ..method.unet import UnetSulciLabeling
        from soma.aimsalgo.sulci import graph_pointcloud
        from soma import aims
        from soma import aimsalgo
        from collections import Counter
        import pandas as pd
        import numpy as np
        import time
        import json

        if self.fix_random_seed:
            import torch
            torch.manual_seed(0)
            try:
                import torch.cudnn
                torch.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            import random
            random.seed(0)
            np.random.seed(0)

        start_time = time.time()
        with open(self.param_file) as f:
            param = json.load(f)
        self.sulci_side_list = param['sulci_side_list']
        method = UnetSulciLabeling(
            self.sulci_side_list, num_filter=64, batch_size=1, cuda=self.cuda)
        method.load(self.model_file)

        dict_sulci = {self.sulci_side_list[i]: i for i in range(len(
            self.sulci_side_list))}
        dict_num = {v: k for k, v in dict_sulci.items()}

        # voxel labeling
        graph = aims.read(self.graph)
        data = extract_data(graph)
        data = {k: np.asarray(v) for k, v in data.items()}

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
