from __future__ import print_function
from sklearn.metrics import recall_score
import numpy as np


def acc_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return len(y_true[y_true == y_pred]) / float(len(y_true))


def bacc_score(y_true, y_pred, labels):
    return recall_score(y_true, y_pred, labels, average='macro')


def esi_score(y_true, y_pred, labels):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp, fp, fn, s = {}, {}, {}, {}
    for ss in labels:
        names_ss = y_pred[y_true == ss]
        labels_ss = y_true[y_pred == ss]
        tp[ss] = float(len(names_ss[names_ss == ss]))
        fp[ss] = float(len(labels_ss[labels_ss != ss]))
        fn[ss] = float(len(names_ss[names_ss != ss]))
        s[ss] = float(len(names_ss))

    sum_s = sum(s.values())
    esi = {}
    for ss in labels:
        if fp[ss] + fn[ss] + 2*tp[ss] != 0 and sum_s != 0:
            esi[ss] = s[ss]/sum_s*(fp[ss]+fn[ss]) / (fp[ss]+fn[ss]+2*tp[ss])
        else:
            esi[ss] = 0

    return sum(esi.values())
