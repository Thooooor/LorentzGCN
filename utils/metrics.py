from time import time

import numpy as np
from torch import Tensor


class AverageRecord(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Metrics class includes Recall@K, NDCG@K
class Metrics:
    def __init__(self, k_list: list, split: str = "valid"):
        self.start_time = time()
        self.split = split
        self.k_list = k_list

        self.y_pred = None
        self.y_true = None

        self.recall = None
        self.ndcg = None

    def update(self, y_pred, y_true):
        if self.y_pred is None:
            self.y_pred = y_pred
            self.y_true = y_true
        else:
            self.y_pred = np.concatenate((self.y_pred, y_pred))
            self.y_true = np.concatenate((self.y_true, y_true))
        self.compute_metrics()

    def compute_metrics(self):
        self.recall = []
        self.ndcg = []
        for k in self.k_list:
            self.recall.append(recall_at_k(self.y_true, self.y_pred, k))
            self.ndcg.append(ndcg_at_k(self.y_true, self.y_pred, k))

    def format_metrics(self):
        result = "{} ".format(self.split)
        for i, k in enumerate(self.k_list):
            result += "Recall@{}: {:.2%} | ".format(k, self.recall[i])
            result += "NDCG@{}: {:.2%} | ".format(k, self.ndcg[i])
        result += "Time: {:.2f}s".format(time() - self.start_time)

        return result

    def to_dict(self):
        result = {}
        for i, k in enumerate(self.k_list):
            result["{} Recall@{}".format(self.split, k)] = self.recall[i]
            result["{} NDCG@{}".format(self.split, k)] = self.ndcg[i]
        return result

    @property
    def metrics(self):
        return self.to_dict()

    def __repr__(self):
        return self.to_dict()

    def __getitem__(self, item):
        return self.metrics[item]


# recall@k
def recall_at_k(y_true: Tensor, y_pred: Tensor, k):
    """

    :param y_true: shape: (batch_size, num_items)
    :param y_pred: shape: (batch_size, num_items)
    :param k:
    :return:
    """
    print(y_true.shape, y_pred.shape)
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    y_pred = y_pred.argsort()[:, -k:]
    y_pred = np.flip(y_pred, axis=1)
    y_true = y_true.argsort()[:, -k:]
    y_true = np.flip(y_true, axis=1)
    recall = 0
    for i in range(k):
        recall += np.sum(y_true[:, i] == y_pred[:, i]) / k
    return recall


# ndcg@k
def ndcg_at_k(y_true: Tensor, y_pred: Tensor, k):
    """
    Normalized discounted cumulative gain (NDCG) at rank K

    :param y_true: shape: (batch_size, num_items)
    :param y_pred:
    :param k:
    :return:
    """
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    y_pred = y_pred.argsort()[:, -k:]
    y_pred = np.flip(y_pred, axis=1)
    y_true = y_true.argsort()[:, -k:]
    y_true = np.flip(y_true, axis=1)
    ndcg = 0
    for i in range(k):
        ndcg += np.sum(y_true[:, i] == y_pred[:, i]) / np.log2(i + 2)
    return ndcg / k
