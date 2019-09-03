from __future__ import print_function
from ..method.snipe import SVMPatternClassification
from capsul.api import Process
from soma import aims

import traits.api as traits
import json


class PatternSVMTraining(Process):
    def __init__(self):
        super(PatternSVMTraining, self).__init__()
        self.add_trait('graphs', traits.List(traits.File(output=False)))
        self.add_trait('pattern', traits.Str(output=False))
        self.add_trait('names_filter', traits.ListStr(output=False))
        self.add_trait('step_1', traits.Bool(
            output=False, optional=True, default=True))
        self.add_trait('step_2', traits.Bool(
            output=False, optional=True, default=True))

        self.add_trait('traindata_file', traits.File(output=True))
        self.add_trait('param_file', traits.File(output=True))

    def _run_process(self):

        # Compute bounding box and labels
        if self.step_1:
            print()
            print('--------------------------------')
            print('---         STEP (1/2)       ---')
            print('--- EXTRACT DATA FROM GRAPHS ---')
            print('--------------------------------')
            print()

            bck_types = ['aims_ss', 'aims_bottom', 'aims_other']
            dict_label, dict_bck, dict_bck_filtered = {}, {}, {}
            dict_searched_pattern = {}
            for gfile in self.graphs:
                print(gfile)
                graph = aims.read(gfile)
                side = gfile[gfile.rfind('/')+1:gfile.rfind('/')+2]
                trans_tal = aims.GraphManip.talairach(graph)
                vs = graph['voxel_size']
                label = 0
                bck, bck_filtered, searched_pattern = [], [], []
                for vertex in graph.vertices():
                    if 'name' in vertex:
                        name = vertex['name']
                        if name.startswith(self.pattern):
                            label = 1
                        fn = sum([1 for n in self.names_filter if name.startswith(n)])
                        fp = 1 if name.startswith(self.pattern) else 0
                        for bck_type in bck_types:
                            if bck_type in vertex:
                                bucket = vertex[bck_type][0]
                                for point in bucket.keys():
                                    fpt = [p * v for p, v in zip(point, vs)]
                                    trpt = list(trans_tal.transform(fpt))
                                    if (side == 'R'):
                                        trpt[0] *= -1
                                    trpt_2mm = [int(round(p/2)) for p in trpt]
                                    bck.append(trpt_2mm)
                                    if fn:
                                        bck_filtered.append(trpt_2mm)
                                    if fp:
                                        searched_pattern.append(trpt_2mm)
                dict_label[gfile] = label
                dict_bck[gfile] = bck
                dict_bck_filtered[gfile] = bck_filtered
                dict_searched_pattern[gfile] = searched_pattern

            # save in parameters file
            param = {'dict_bck': dict_bck,
                     'dict_bck_filtered': dict_bck_filtered,
                     'dict_label': dict_label,
                     'dict_searched_pattern': dict_searched_pattern}
            with open(self.traindata_file, 'w') as f:
                json.dump(param, f)
        else:
            with open(self.traindata_file) as f:
                param = json.load(f)
            dict_bck = param['dict_bck']
            dict_bck_filtered = param['dict_bck_filtered']
            dict_searched_pattern = param['dict_searched_pattern']
            dict_label = param['dict_label']

        method = SVMPatternClassification(
            pattern=self.pattern, names_filter=self.names_filter,
            dict_bck=dict_bck, dict_bck_filtered=dict_bck_filtered,
            dict_searched_pattern=dict_searched_pattern, dict_label=dict_label)

        # Inner cross validation - fix learning rate / momentum
        if self.step_2:
            print()
            print('---------------------------')
            print('---      STEP (2/2)     ---')
            print('--- FIX HYPERPARAMETERS ---')
            print('---------------------------')
            print()
            method.find_hyperparameters(self.graphs, self.param_file)
