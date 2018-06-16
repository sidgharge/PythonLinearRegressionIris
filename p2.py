from numpy.core.multiarray import dtype
from sklearn import datasets
import numpy as np


def calc_cost(w: np.core.multiarray, b: int, x: np.core.multiarray, y: np.core.multiarray):
    a = np.matmul(np.transpose(w), x) + b
    return np.power((a - y), 2) / (2 * np.shape(y)[0])


def cal_dau_w(w: np.core.multiarray, b: int, x: np.core.multiarray, y: np.core.multiarray):
    a = np.matmul(np.transpose(w), x) + b
    return np.matmul(x, np.transpose(a - y)) / np.shape(y)[1]


def cal_dau_b(w: np.core.multiarray, b: int, x: np.core.multiarray, y: np.core.multiarray):
    a = np.matmul(np.transpose(w), x) + b
    return np.sum((a - y)) / np.shape(y)[1]
    # return (a - y) / np.shape(y)[1]


def update_w(w: np.core.multiarray, dw: np.core.multiarray):
    alpha = 0.01
    return w - alpha * dw


def update_b(b, db):
    alpha = 0.01
    return b - alpha * db


iris = datasets.load_iris()
data = iris["data"]
print(np.shape([data[:, 3]]))
y = np.array([data[:, 2]])
x = np.transpose(data[:, 0:2])
x = np.append(x, ([data[:, 3]]), axis=0)
w = np.zeros((x.shape[0], 1), dtype=float)
b = 0

for i in range(10000):
    cost = calc_cost(w, b, x, y)
    dw = cal_dau_w(w, b, x, y)
    db = cal_dau_b(w, b, x, y)
    w = update_w(w, dw)
    b = update_b(b, db)

print('New w: \n', w)
print('New b: \n', b)
