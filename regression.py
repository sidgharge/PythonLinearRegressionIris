import numpy as np


class Regression:

    def __init__(self, alpha, iterations, xs: np.core.multiarray, ys: np.core.multiarray):
        self.alpha = alpha
        self.xs = xs[0:int(0.8*len(xs))]
        self.ys = ys[0:int(0.8*len(ys))]
        self.w = 0
        self.b = 0
        self.dw = 0
        self.db = 0
        self.cost = 0
        self.iterations = iterations
        self.xs_training = xs[int(len(xs) * 0.8):len(xs)]
        self.ys_training = ys[int(len(ys) * 0.8):len(ys)]

    def calc_cost(self):
        temp = 0
        for x, y in zip(self.xs, self.ys):
            temp += np.power(((self.w * x + self.b) - y), 2)
        self.cost = temp / (2 * len(self.xs))

    def cal_dau_w(self):
        temp = 0
        for x, y in zip(self.xs, self.ys):
            temp += (self.w * x + self.b - y) * x
        self.dw = temp / len(self.xs)

    def cal_dau_b(self):
        temp = 0
        for x, y in zip(self.xs, self.ys):
            temp += (self.w * x + self.b - y)
        self.db = temp / len(self.xs)

    def update_w(self):
        self.w = self.w - (self.alpha * self.dw)

    def update_b(self):
        self.b = self.b - (self.alpha * self.db)

    def start_training(self):
        for i in range(self.iterations):
            self.calc_cost()

            self.cal_dau_w()

            self.cal_dau_b()

            self.update_w()

            self.update_b()
        return self.w, self.b

    def predict_y(self, x):
        return self.w * x + self.b

    def check_accuracy(self):
        accuracy = 0
        for x, y in zip(self.xs_training, self.ys_training):
            y_pred = self.predict_y(x)
            accuracy += abs((y_pred - y) / y_pred)
        accuracy /= len(self.xs_training)
        return 1 - accuracy


