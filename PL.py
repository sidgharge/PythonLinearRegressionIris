from sklearn import datasets
import numpy as np
from multifeaturesregression import MultiFeaturesRegression as reg


iris = datasets.load_iris()
data = iris["data"]
print(np.shape([data[:, 3]]))
y = np.array([data[:, 2]])
x = np.transpose(data[:, 0:2])
x = np.append(x, ([data[:, 3]]), axis=0)
w = np.zeros((x.shape[0], 1), dtype=float)

r = reg(0.01, 10000, x, y)
print(r.start_training())
