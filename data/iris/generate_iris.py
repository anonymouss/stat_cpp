#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
data = iris.data[:100]
label = iris.target[:100]

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
y_train = np.array([1 if (i == 1).all() else -1 for i in y_train])
y_test = np.array([1 if (i == 1).all() else -1 for i in y_test])

np.savetxt('X_train', X_train, fmt='%.1f')
np.savetxt('X_test', X_test, fmt='%.1f')
np.savetxt('y_train', y_train, fmt='%d')
np.savetxt('y_test', y_test, fmt='%d')
