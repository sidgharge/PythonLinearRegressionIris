from sklearn import datasets
from regression import Regression as reg

iris = datasets.load_iris()
data = iris["data"]

r = reg(0.01, 10000, data[:, 3], data[:, 2])

res = r.start_training()

print(res)

accuracy = r.check_accuracy()

print(accuracy)
