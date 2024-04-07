import numpy as np


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        w = np.array([w])
        y = np.array([y])
        M = X @ w.T
        Q = (np.logaddexp(0, -y.T * M)).sum() / y.shape[1]
        l2 = 0.5 * self.l2_coef * (w.T ** 2).sum()
        return Q + l2

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        w = np.array([w])
        y = np.array([y])
        M = X @ w.T
        denominator = (1 + np.exp(y.T * M))
        if isinstance(X, np.ndarray):
            ans = -1 / y.shape[1] * (1 / denominator * (y.T) * X).sum(axis=0)
        else:
            ans = -1 / y.shape[1] * (X.multiply(1 / denominator * (y.T)).toarray()).sum(axis=0)
        return np.array([ans]).T + self.l2_coef * w.T
