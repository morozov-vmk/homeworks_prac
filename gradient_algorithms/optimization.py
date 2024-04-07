import time
import numpy as np
import scipy


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


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function != 'binary_logistic':
            raise TypeError
        self.loss_function = BinaryLogistic(kwargs['l2_coef'])
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs

    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        self.w = np.array([w_0]).T
        history = {'time': [0], 'func': [self.loss_function.func(X, y, self.w.T[0])]}
        self.w = np.array([w_0]).T
        last_time = time.time()
        for i in range(1, self.max_iter + 1):
            if i != 1 and abs(history['func'][-1] - history['func'][-2]) <= self.tolerance:
                break
            grad = self.loss_function.grad(X, y, self.w.T[0])
            self.w -= self.step_alpha / i ** self.step_beta * grad
            history["func"].append(self.loss_function.func(X, y, self.w.T[0]))
            history["time"].append(time.time() - last_time)
            last_time = time.time()
        if trace:
            return history

    def predict(self, X):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        return 2 * (X @ self.w >= 0.5) - 1

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        a = scipy.special.expit(X @ self.w)
        return np.concatenate((1 - a, a), axis=1)

    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        return self.loss_function.func(X, y, self.w)

    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        return self.loss_function.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.w


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
        self, loss_function, batch_size, step_alpha=1, step_beta=0,
        tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """
        if loss_function != 'binary_logistic':
            raise TypeError
        self.loss_function = BinaryLogistic(kwargs['l2_coef'])
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.kwargs = kwargs

    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        self.w = np.array([w_0]).T
        self.prev_w = np.array([w_0]).T
        history = {'time': [0],
                   'func': [self.loss_function.func(X, y, self.w.T[0])],
                   'epoch_num': [0],
                   'weights_diff': [0]}
        last_time = time.time()
        prev_epoch_num, epoch_num = 0, 0
        for i in range(1, self.max_iter + 1):
            if (len(history['func']) > 1) and abs(history['func'][-1] - history['func'][-2]) <= self.tolerance:
                break
            new_ind = np.arange(X.shape[0])
            np.random.shuffle(new_ind)
            X_new = X[new_ind]
            y_new = y[new_ind]
            for j in range(0, X.shape[0], self.batch_size):
                X_batch = X_new[j:j + self.batch_size]
                y_batch = y_new[j:j + self.batch_size]
                grad = self.loss_function.grad(X_batch, y_batch, self.w.T[0])
                self.w -= self.step_alpha / i ** self.step_beta * grad
                epoch_num += self.batch_size / X.shape[0]
                if not trace:
                    continue
                if epoch_num - prev_epoch_num >= log_freq:
                    history["epoch_num"].append(epoch_num)
                    prev_epoch_num = epoch_num
                    history["time"].append(time.time() - last_time)
                    last_time = time.time()
                    history["func"].append(self.loss_function.func(X, y, self.w.T[0]))
                    history["weights_diff"].append(((self.prev_w.T[0] - self.w.T[0]) ** 2).sum())
                    self.prev_w = self.w.copy()
        if trace:
            return history
