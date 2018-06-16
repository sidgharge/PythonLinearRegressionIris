import numpy as np


class MultiFeaturesRegression:

    def __init__(self, alpha, iterations, xs: np.core.multiarray, ys: np.core.multiarray):
        self.alpha = alpha
        self.x = xs[:, 0:int(0.8*np.shape(xs)[1])]
        self.y = ys[:, 0:int(0.8*np.shape(ys)[1])]
        self.w = np.zeros((self.x.shape[0], 1), dtype=float)
        self.b = 0
        self.dw = np.zeros(self.w.shape, dtype=float)
        self.db = 0
        self.cost = 0
        self.iterations = iterations
        self.xs_training = xs[:, int(np.shape(xs)[1] * 0.8):np.shape(xs)[1]]
        self.ys_training = ys[:, int(np.shape(ys)[1] * 0.8):np.shape(ys)[1]]

    def calc_cost(self):
        a = np.matmul(np.transpose(self.w), self.x) + self.b
        self.cost = np.power((a - self.y), 2) / (2 * np.shape(self.y)[0])

    def cal_dau_w(self):
        a = np.matmul(np.transpose(self.w), self.x) + self.b
        self.dw = np.matmul(self.x, np.transpose(a - self.y)) / np.shape(self.y)[1]

    def cal_dau_b(self):
        a = np.matmul(np.transpose(self.w), self.x) + self.b
        self.db = np.sum((a - self.y)) / np.shape(self.y)[1]

    def update_w(self):
        self.w = self.w - self.alpha * self.dw

    def update_b(self):
        self.b = self.b - self.alpha * self.db

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
        y_pred = np.matmul(np.transpose(self.w), self.xs_training) + self.b
        accuracy = abs(self.ys_training - y_pred) / y_pred
        return 1 - np.sum(accuracy) / accuracy.shape[1]
