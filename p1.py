from sklearn import datasets
import numpy as np


def calc_cost(w: int, b: int, xs: np.core.multiarray, ys: np.core.multiarray):
    temp = 0
    for x, y in zip(xs, ys):
        temp += np.power(((w * x + b) - y), 2)
    return temp / (2 * len(xs))


def cal_dau_w(w: int, b: int, xs: np.core.multiarray, ys: np.core.multiarray):
    temp = 0
    for x, y in zip(xs, ys):
        temp += (w * x + b - y) * x
    return temp / len(xs)


def cal_dau_b(w: int, b: int, xs: np.core.multiarray, ys: np.core.multiarray):
    temp = 0
    for x, y in zip(xs, ys):
        temp += (w * x + b - y)
    return temp / len(xs)


def update_w(w: int, dw: int):
    # let alpha = 0.01
    alpha = 0.01
    return w - (alpha * dw)


def update_b(b: int, db: int):
    # let alpha = 0.01
    alpha = 0.01
    return b - (alpha * db)


iris = datasets.load_iris()
data = iris["data"]

w = 0
b = 0
for i in range(10000):
    cost = calc_cost(w, b, data[:, 3], data[:, 2])

    dw = cal_dau_w(w, b, data[:, 3], data[:, 2])

    db = cal_dau_b(w, b, data[:, 3], data[:, 2])

    w = update_w(w, dw)

    b = update_b(b, db)

print('New w: ', w, ' New b: ', b)

# iris = datasets.load_iris()
# data = iris["data"]
# print(data[:, 2])
# print(data[:, 3])
