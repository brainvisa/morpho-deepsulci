from __future__ import print_function
from __future__ import absolute_import
from sklearn.metrics import recall_score


def balanced_accuracy(y_true, y_pred, labels=None):
    '''
    balanced_accuracy
    '''
    return recall_score(y_true, y_pred, labels=labels, average='macro')
