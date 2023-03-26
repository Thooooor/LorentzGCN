from time import time
import numpy as np


class AverageRecord(object):
    """
    Compute and store the average and current value
    """

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


class Metrics:
    """
    Compute metrics for evaluation
    """

    def __init__(self, k_list: list, split: str = "valid"):
        self.start_time = time()
        self.split = split
        self.k_list = k_list

        self.y_pred = None
        self.y_true = None

        self.recall = None
        self.ndcg = None

    def update(self, y_pred, y_true):
        """
        Update y_pred and y_true
        :param y_pred:
        :param y_true:
        :return:
        """
        if self.y_pred is None:
            self.y_pred = np.array(y_pred)
            self.y_true = np.array(y_true)
        else:
            self.y_pred = np.concatenate((self.y_pred, np.array(y_pred)), axis=0)
            self.y_true = np.array(self.y_true.tolist() + y_true)

    def compute_metrics(self):
        """Compute recall and ndcg"""
        self.get_pred_label()
        self.recall = []
        self.ndcg = []
        for k in self.k_list:
            self.recall.append(recall_at_k(self.y_true, self.y_pred, k))
            self.ndcg.append(ndcg_at_k(self.y_true, self.y_pred, k))

    def get_pred_label(self):
        """Convert prediction to label"""
        for i in range(self.y_pred.shape[0]):
            self.y_pred[i] = np.array(list(map(lambda x: 1 if x in self.y_true[i] else 0, self.y_pred[i])))

    def format_metrics(self):
        """Format metrics to string"""
        result = ""
        for i, k in enumerate(self.k_list):
            result += "Recall@{}: {:.2} | ".format(k, self.recall[i])
            result += "NDCG@{}: {:.2} | ".format(k, self.ndcg[i])
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
        return self.format_metrics()

    def __getitem__(self, item):
        return self.metrics[item]


def recall_at_k(y_true, y_pred, k):
    """
    Recall at rank K

    :param y_true:
    :param y_pred: shape: (num_users, top_k)
    :param k:
    :return:
    """
    y_pred_right = y_pred[:, :k].sum(axis=1)

    recall_n = np.array([len(y_true[i]) for i in range(len(y_true))])
    recall = np.mean(y_pred_right / recall_n)

    return recall


def ndcg_at_k(y_true, y_pred, k):
    """
    Normalized discounted cumulative gain (NDCG) at rank K

    :param y_true:
    :param y_pred: shape: (num_users, top_k)
    :param k:
    :return:
    """
    y_pred = y_pred[:, :k]
    true_matrix = np.zeros((y_pred.shape[0], k))

    for i, items in enumerate(y_true):
        length = k if len(items) > k else len(items)
        true_matrix[i, :length] = 1

    idcg = np.sum(true_matrix / np.log2(np.arange(2, true_matrix.shape[1] + 2)), axis=1)
    dcg = y_pred * (1.0 / np.log2(np.arange(2, y_pred.shape[1] + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.

    return np.mean(ndcg)
