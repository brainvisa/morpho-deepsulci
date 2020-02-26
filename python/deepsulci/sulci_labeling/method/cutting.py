from __future__ import print_function
from __future__ import absolute_import
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import radius_neighbors_graph
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from collections import Counter
import numpy as np
from six.moves import range
from six.moves import zip


def cutting(y_scores, y_vert, bck, threshold, vs=1.):
    '''
    Cut elementary fold according to voxel-wise classification scores
    '''

    # unique points
    # TODO. Optimizer parce que cest trop trop long
    bck = np.asarray(bck)
    conv = np.zeros(len(bck), dtype=int)
    conv -= 1
    ubck = []
    for p, i in zip(bck, range(len(bck))):
        if conv[i] < 0:
            conv[np.all(bck == p, axis=1)] = len(ubck)
            ubck.append(p)
    ubck = np.asarray(ubck)
    inv_conv = [[] for i in ubck]
    for i, j in zip(range(len(conv)), conv):
        inv_conv[j].append(i)

    y_vert = np.asarray(y_vert)
    y_uvert = np.asarray([Counter(y_vert[inv_conv[i]]).most_common()[0][0] for i in range(len(inv_conv))])
    y_scores = np.asarray(y_scores)
    y_uscores = np.asarray([y_scores[inv_conv[i][0]] for i in range(len(inv_conv))])

    y_upred = np.zeros(len(y_uvert), dtype=object)
    y_pred = np.zeros(len(y_vert), dtype=object)
    for v in set(y_vert):
        l = y_uscores[y_uvert == v].mean(axis=0).argmax()
        y_upred[y_uvert == v] = l
        y_pred[y_vert == v] = l

        vuscores = y_uscores[y_uvert == v]
        vupred = y_upred[y_uvert == v]
        vupoints = ubck[y_uvert == v]
        vuclusters = np.zeros(len(vupred))
        num_clusters = 0

        if len(vuscores) > 4:
            test_cut = True
            while test_cut:
                test_cut = False
                for cl in set(vuclusters):
                    cl_points = vupoints[vuclusters == cl]
                    cl_scores = vuscores[vuclusters == cl]
                    cl_clusters = vuclusters[vuclusters == cl]
                    cl_pred = vupred[vuclusters == cl]
                    connectivity = radius_neighbors_graph(
                        cl_points, radius=1.8*vs, include_self=False)

                    ward = AgglomerativeClustering(
                        n_clusters=2, connectivity=connectivity,
                        linkage='ward').fit(cl_scores)
                    label = ward.labels_
                    if Counter(label).most_common()[-1][1] > 1:
                        ch_score = metrics.calinski_harabaz_score(
                            cl_scores, label)
                        if ch_score > threshold:
                            test_cut = True
                            lj_list = []
                            for j in set(label):
                                lj = cl_scores[label == j].mean(axis=0).argmax()
                                cl_pred[label == j] = lj
                                cl_clusters[label == j] = num_clusters + 1
                                num_clusters += 1
                                lj_list.append(lj)
                            vupred[vuclusters == cl] = cl_pred
                            vuclusters[vuclusters == cl] = cl_clusters  
#                            print('CUTTING')
            y_upred[y_uvert == v] = vupred
            if len(set(vupred)) > 1:
                vconv = conv[y_vert == v]
                vbck = bck[y_vert == v]
                vpred = y_upred[vconv]
                neigh = NearestNeighbors()
                neigh.fit(vbck[[True if vi in set(vupred) else False for vi in vpred]])
                for i in range(len(vpred)):
                    if vpred[i] not in set(vupred):
                        ind = neigh.kneighbors([vbck[i]], 1,
                                               return_distance=False)
                        vpred[i] = vpred[ind[0][0]]
                y_pred[y_vert == v] = vpred

    y_pred = np.asarray(y_pred)
    return y_pred
