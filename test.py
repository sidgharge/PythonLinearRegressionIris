from sklearn import datasets
import numpy as np

a = np.arange(10).reshape((2, 5))
b = np.arange(2).reshape((2, 1))
c = np.append(a, b, axis=1)
print(a[:, 0:3])
