from sklearn import datasets
import numpy as np
from multifeaturesregression import MultiFeaturesRegression as reg


iris = datasets.load_iris()
data = iris["data"]
y = np.array([data[:, 2]])
x = np.transpose(data[:, 0:2])
x = np.append(x, ([data[:, 3]]), axis=0)
w = np.zeros((x.shape[0], 1), dtype=float)

r = reg(0.01, 10000, x, y)
w_and_b = r.start_training()
print("Value of w: ", w_and_b[0])
print("Value of b: ", w_and_b[1])

print("Accuracy is: ", r.check_accuracy())
