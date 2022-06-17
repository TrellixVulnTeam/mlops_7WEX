import torch
import numpy as np
from sklearn import metrics


def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = 0
    correct_k = correct[:1].view(-1).float().sum(0)
    res = correct_k.mul_(100.0 / batch_size)
    return res

def confusion_matrix(preds, labels, conf_matrix):
    preds = np.argmax(preds.to('cpu').detach().numpy(),1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] +=1
    return conf_matrix

def metrics_cal(conf_matrix):
    num_classes = conf_matrix.shape[0]
    recall_lst = []
    precision_lst = []
    speciticity_lst = []

    for c in range(num_classes):
        precision_lst.append(conf_matrix[c,c]/conf_matrix[c].sum())
        recall_lst.append(conf_matrix[c,c]/conf_matrix[:,c].sum())
        speciticity_lst.append(np.delete(np.delete(conf_matrix,c,0),c,1).sum()/np.delete(conf_matrix,c,0).sum())
    acc = np.diag(conf_matrix).sum()/conf_matrix.sum()
    recall = sum(recall_lst)/num_classes
    precision = sum(precision_lst)/num_classes
    speciticity = sum(speciticity_lst)/num_classes
    f_1 = 2*((precision*recall)/(precision+recall))

    return acc, recall, precision, speciticity, f_1
