'''
Error Computation Module
'''

from __future__ import print_function
from __future__ import absolute_import
from soma import aims, aimsalgo
from capsul.api import Process
import traits.api as traits
import pandas as pd
import numpy as np
from six.moves import range
from six.moves import zip


class ErrorComputation(Process):
    '''
    Process to compute the Similarity Index Error rate (ESI) and the local
    error rate (Elocal) described in (Perrot et al. 2011).
    In order to evaluate the impact of the pipeline variability, the error
    rates can be computed on several graphs (extracted from the same MRI scan)
    automatically labelled. These graphs are compared to the manually labelled
    graph (true_graph).
    
    '''

    def __init__(self):
        super(ErrorComputation, self).__init__()
        self.add_trait('t1mri', traits.File(
            output=False, desc='MRI scan'))
        self.add_trait('true_graph', traits.File(
            output=False, desc='corresponding graph manually labelled'))
        self.add_trait('labelled_graphs', traits.List(
            traits.File(output=False),
            desc='corresponding set of graphs automatically labelled'))
        self.add_trait('sulci_side_list', traits.List(
            traits.Str(output=False),
            desc='list of sulci (e.g. S.C._right) considered to compute the'
                 ' error rates. It is not supposed to contain the labels'
                 ' "unknown", "ventricle_left" and "ventricle_right".'))

        self.add_trait('error_file', traits.File(
            output=True,
            desc='file (.csv) storing the error rates for each labelled graph'))

    def _run_process(self):
        # Compute voronoi
        mri = aims.read(self.t1mri)
        true_graph = aims.read(self.true_graph)
        vs = true_graph['voxel_size']
        vvol = vs[0]*vs[1]*vs[2]
        true_bck, true_names, _ = extract_data(true_graph)
        nlist = list(set(true_names))
        dnames = {k: v+1 for k, v in zip(nlist, range(len(nlist)))}
        dnum = {v+1: k for k, v in zip(nlist, range(len(nlist)))}
        fm = aims.FastMarching()
        vol = aims.Volume_S16(mri.getSizeX(), mri.getSizeY(), mri.getSizeZ())
        vol.fill(0)
        for p, n in zip(true_bck, true_names):
            vol[p[0], p[1], p[2]] = dnames[n]

        fm.doit(vol, [0], list(dnames.values()))
        vor = fm.voronoiVol()

        # Compute error rates
        re = pd.DataFrame(index=[str(g) for g in self.labelled_graphs])
        for gfile in self.labelled_graphs:
            graph = aims.read(gfile)
            bck, _, labels = extract_data(graph)
            y_pred = [vor[int(round(p[0])),
                          int(round(p[1])),
                          int(round(p[2]))][0] for p in bck]
            names = np.asarray([dnum[n] for n in y_pred])

            for ss in self.sulci_side_list:
                names_ss = labels[names == ss]
                labels_ss = names[labels == ss]

                re.ix[gfile, 'TP_'+str(ss)] = float(len(names_ss[names_ss == ss]))*vvol
                re.ix[gfile, 'FP_'+str(ss)] = float(len(labels_ss[labels_ss != ss]))*vvol
                re.ix[gfile, 'FN_'+str(ss)] = float(len(names_ss[names_ss != ss]))*vvol
                re.ix[gfile, 's_'+str(ss)] = float(len(names_ss))*vvol

            sum_s = sum([re.ix[gfile, 's_'+str(ss)] for ss in self.sulci_side_list])
            for ss in self.sulci_side_list:
                FP = re.ix[gfile, 'FP_'+str(ss)]
                FN = re.ix[gfile, 'FN_'+str(ss)]
                VP = re.ix[gfile, 'TP_'+str(ss)]
                s = re.ix[gfile, 's_'+str(ss)]
                if FP + FN + 2*VP != 0:
                    re.ix[gfile, 'ESI_'+str(ss)] = s/sum_s * (FP + FN) / (FP + FN + 2*VP)
                    re.ix[gfile, 'Elocal_'+str(ss)] = s/sum_s * (FP + FN) / (FP + FN + VP)
                else:
                    re.ix[gfile, 'ESI_'+str(ss)] = 0
                    re.ix[gfile, 'Elocal_'+str(ss)] = 0

            re.ix[gfile, 'ESI'] = sum([re.ix[gfile, 'ESI_'+str(ss)]
                                      for ss in self.sulci_side_list])

        re.to_csv(self.error_file)

        print('Mean ESI: %.3f' % re['ESI'].mean())
        print('Max ESI: %.3f' % re['ESI'].max())
        print()
        for ss in self.sulci_side_list:
            print('%s Elocal mean: %.3f, max: %.3f' %
                  (ss, re['Elocal_'+str(ss)].mean(),
                   re['Elocal_'+str(ss)].max()))


def extract_data(graph):
    bck_types = ['aims_ss', 'aims_bottom', 'aims_other']
    names, labels, bck = [], [], []
    for vertex in graph.vertices():
        if 'name' in vertex:
            name = vertex['name']
        else:
            name = 'unknown'
        if 'label' in vertex:
            label = vertex['label']
        else:
            label = 'unknown'
        for bck_type in bck_types:
            if bck_type in vertex:
                bucket = vertex[bck_type][0]
                for point in bucket.keys():
                    bck.append(list(point))
                    names.append(name)
                    labels.append(label)
    return np.asarray(bck), np.asarray(names), np.asarray(labels)
