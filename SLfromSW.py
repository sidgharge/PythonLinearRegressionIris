from sklearn import datasets
from regression import Regression as reg

iris = datasets.load_iris()
data = iris["data"]

r = reg(0.1, 10000, data[:, 1], data[:, 0])

res = r.start_training()

print(1 - r.cost)

accuracy = r.check_accuracy()

print(accuracy)
