import numpy as np


def grad_num(f, w, delta):
    answer = []
    for i in range(w.shape[0]):
        tmp = np.zeros((w.shape[0],))
        tmp[i] = delta
        answer.append((f(w + tmp) - f(w)) / delta)
    return answer

def grad(X, y, w, l2_coef):
    w = np.array([w])
    y = np.array([y])
    M = X @ w.T
    denominator = (1 + np.exp(y.T * M))
    if isinstance(X, np.ndarray):
        ans = -1 / y.shape[1] * (1 / denominator * (y.T) * X).sum(axis=0)
    else:
        ans = -1 / y.shape[1] * (X.multiply(1 / denominator * (y.T)).toarray()).sum(axis=0)
    return (np.array([ans]).T + l2_coef * w.T).T[0]

def func(w):
    global y_train
    global X_train_matrix
    X = X_train_matrix
    w = np.array([w])
    y = np.array([y_train])
    M = X @ w.T
    Q = (np.logaddexp(0, -y.T * M)).sum() / y.shape[1]
    l2 = 0.5 * (w.T ** 2).sum()
    return Q + l2