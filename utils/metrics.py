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

    def compute_metrics(self, pred_matrix, ture_dict, user_item_csr):
        """Compute recall and ndcg"""
        top_k = max(self.k_list)
        pred_matrix[user_item_csr.nonzero()] = np.NINF
        
        ind = np.argpartition(pred_matrix, -top_k)
        ind = ind[:, -top_k:]
        arr_ind = pred_matrix[np.arange(len(pred_matrix))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(pred_matrix)), ::-1]
        pred_list = ind[np.arange(len(pred_matrix))[:, None], arr_ind_argsort]
        all_ndcg = ndcg_func([*ture_dict.values()], pred_list)
        
        self.recall = [recall_at_k(ture_dict, pred_list, k) for k in self.k_list]
        self.ndcg = [all_ndcg[k-1] for k in self.k_list]

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
    sum_recall = 0.0
    num_users = len(y_true)
    true_users = 0
    
    for i, v in y_true.items():
        true_set = set(v)
        pred_set = set(y_pred[i][:k])
        if len(true_set) != 0:
            sum_recall += len(true_set & pred_set) / float(len(true_set))
            true_users += 1
            
    assert num_users == true_users
    return sum_recall / true_users


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


def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)
