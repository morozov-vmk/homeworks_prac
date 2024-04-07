import numpy as np
from sklearn.neighbors import NearestNeighbors
import math


def euclidean_distance(matr1, matr2):
    ans = (- 2) * matr1.dot(matr2.T)
    ans += (matr1**2).sum(1, keepdims=True)
    ans += (matr2**2).sum(1)
    return np.sqrt(ans)


def cosine_distance(matr1, matr2):
    ans = -matr1.dot(matr2.T)
    ans[ans == 0] = 1
    b = np.sqrt((matr1**2).sum(1, keepdims=True))
    b[b == 0] = 1
    ans /= b
    b = np.sqrt((matr2**2).sum(1))
    b[b == 0] = 1
    # for 0-vector scalar product = 0 and norma = 0
    # 0/0 eqv 1
    ans /= b
    return np.ones([matr1.shape[0], matr2.shape[0]]) + ans


class KNNClassifier:
    def __init__(self, k=5, strategy='brute', metric='euclidean',
                 weights=False, test_block_size=1):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size

    def fit(self, X, y):
        self.X = X
        self.y = y
        if self.strategy == 'my_own':
            return
        self.neigh = NearestNeighbors(n_neighbors=self.k,
                                      algorithm=self.strategy,
                                      metric=self.metric)
        self.neigh.fit(X, y)

    def find_kneighbors(self, X, return_distance):
        if self.strategy != 'my_own':
            return self.neigh.kneighbors(X, return_distance=return_distance)
        if self.strategy == 'my_own':
            if self.metric == 'cosine':
                dist = cosine_distance(X, self.X)
            else:
                dist = euclidean_distance(X, self.X)
            ind = np.argpartition(dist, np.arange(self.k + 1), axis=1)
            # sort weights (from 0 to k) and take first k
            # sort and argsort take many memory, because sort all elements
            # insteed first k metted, so I use argpartition, take_along_axis
            if return_distance:
                return (np.take_along_axis(dist, ind[:, :(self.k)], axis=1),
                        ind[:, :(self.k)])
            return (ind[:, :(self.k)])

    def predict(self, X):
        (weights, ind) = self.find_kneighbors(X, True)
        if self.weights:
            weights = 1 / (weights + 0.00001)
        else:
            weights = np.ones([X.shape[0], self.k])
        new_weights = np.ndarray([X.shape[0], self.k, 1])
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                new_weights[i][j][0] = weights[i][j]
        new_matr_y = np.ndarray([ind.shape[0], ind.shape[1], 1])
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                x = ((i * ind.shape[1] + j) // self.y[ind].shape[1])
                y = ((i * ind.shape[1] + j) % self.y[ind].shape[1])
                new_matr_y[i][j][0] = self.y[ind][x][y]
        arg_sum = (new_matr_y == np.unique(self.y)) * new_weights
        summing = np.sum(arg_sum, axis=1)
        ans_ind = np.argmax(summing, axis=1)
        return np.unique(self.y)[ans_ind]
# such reshape needs to take argmax from each element


def kfold(n, n_folds):
    answer = []
    all_coor = np.arange(n)
    count_short_val = n_folds - n % n_folds
    len_long_val = n // n_folds + 1
    len_short_val = n // n_folds
    count_long_val = (n - len_short_val * count_short_val) // len_long_val
    for i in range(count_long_val):
        arr1 = all_coor[: i * len_long_val]
        arr1 = np.append(arr1, all_coor[(i + 1) * len_long_val:])
        arr2 = all_coor[i * len_long_val: (i + 1) * len_long_val]
        answer.append((arr1, arr2))
    offset = len_long_val * count_long_val
    for i in range(count_short_val):
        arr1 = all_coor[: i * len_short_val + offset]
        arr1 = np.append(arr1, all_coor[(i + 1) * len_short_val + offset:])
        left = i * len_short_val + offset
        right = (i + 1) * len_short_val + offset
        arr2 = all_coor[left: right]
        answer.append((arr1, arr2))
    return answer


def knn_cross_val_score(X, y, k_list, score="accuracy", cv=None,
                        **kwargs):
    if score != 'accuracy':
        return
    answer = {}
    if not cv:
        cv = kfold(X.shape[0], 5)
    knn = KNNClassifier(k=k_list[len(k_list) - 1], **kwargs)
    # I take last k, because if make new KNNClassifier for each k => TL
    # Last k suit for each k
    i_cv = 0
    for train_i, test_i in cv:
        knn.fit(X[train_i], y[train_i])
        len_y = len(y[test_i])
        (dist, ind) = knn.find_kneighbors(X[test_i], True)
        all_names_in_train = y[train_i]
        names = all_names_in_train[ind]
        for k in k_list:
            if i_cv == 0:
                answer[k] = np.ndarray([len(cv)])
            dist_k = dist[:, :k]
            best_names = np.ndarray([names.shape[0]])
            cons_names = names[:, :k]
            j = 0
            for names_j in cons_names:
                if knn.weights:
                    weig = (1 / (dist_k[j] + 0.00001))
                else:
                    weig = np.ones(dist_k[j].shape)
                weights_res = np.bincount(names_j, weights=weig)
                name_of_max = np.argsort(weights_res)[-1]
                best_names[j] = name_of_max
                j += 1
            tmp_ans = 0
            for i in range(len_y):
                if best_names[i] == (y[test_i])[i]:
                    tmp_ans += 1
            (answer[k])[i_cv] = tmp_ans / len_y
        i_cv += 1
    return answer
