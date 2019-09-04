from __future__ import print_function
from soma import aims
from capsul.api import Process
import traits.api as traits
import pandas as pd


class ErrorComputation(Process):
    def __init__(self):
        super(ErrorComputation, self).__init__()
        self.add_trait('irm', traits.File(output=False))
        self.add_trait('true_graph', traits.File(output=False))
        self.add_trait('labeled_graphs',
                       traits.List(traits.File(output=False)))
        self.add_trait('sulci_side_list',
                       traits.List(traits.Str(output=False)))

        self.add_trait('error_rates', traits.File(output=True))

    def _run_process(self):
        # Compute voronoi
        irm = aims.read(self.irm)
        true_graph = aims.read(self.true_graph)
        vs = true_graph['voxel_size']
        vvol = vs[0]*vs[1]*vs[2]
        true_bck, true_names, _ = extract_data(true_graph)
        nlist = list(set(true_names))
        dnames = {k: v+1 for k, v in zip(nlist, range(len(nlist)))}
        dnum = {v+1: k for k, v in zip(nlist, range(len(nlist)))}
        fm = aims.FastMarching()
        vol = aims.Volume_S16(irm.getSizeX(), irm.getSizeY(), irm.getSizeZ())
        vol.fill(0)
        for p, n in zip(true_bck, true_names):
            vol[p[0], p[1], p[2]] = dnames[n]

        fm.doit(vol, [0], list(dnames.values()))
        vor = fm.voronoiVol()

        # Compute error rates
        re = pd.DataFrame(index=self.labeled_graphs)
        for gfile in self.labeled_graphs:
            graph = aims.read(gfile)
            bck, _, labels = extract_data(graph)
            y_pred = [vor[int(round(p[0])),
                          int(round(p[1])),
                          int(round(p[2]))][0] for p in bck]
            names = [dnum[n] for n in y_pred]

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

        re.to_csv(self.erro_rates)

        print('Mean ESI: %r' % re['ESI'].mean())
        print('Max ESI: %r' % re['ESI'].max())
        print()
        for ss in self.sulci_side_list:
            print('%s Elocal mean: %r, max: %r' %
                  (ss, re['Elocal_'+str(ss)].mean(),
                   re['Elocal_'+str(ss)].max()))


def extract_data(graph):
    bck_types = ['aims_ss', 'aims_bottom', 'aims_other']
    names, labels, bck = [], []
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
    return bck, names, labels
