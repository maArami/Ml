from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from LinearRegressin import LinearRegression

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], y, color='b', marker='o', s=30)

LR = LinearRegression()
LR.fit(X_train, y_train)
y_test_predict = LR.predict(X_test)
Eout = np.mean( (y_test_predict - y_test)**2 )

xmin = min(X_train[:,0])
xmax = max(X_train[:,0])
ymax = xmax*LR.weights[0] + LR.weights[1]
ymin = xmin*LR.weights[0] +LR.weights[1]

plt.plot([xmin, xmax], [ymin, ymax], color='r')
plt.scatter(X_train[:,0], y_train, color='b' , marker='o', s=10)
plt.scatter(X_test[:,0], y_test, color='y', marker='o', s=20 )