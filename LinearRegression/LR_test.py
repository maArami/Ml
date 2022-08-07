from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

from LinearRegressin import LinearRegression
from LinearRegressin import LinearRegression_GD

X, y = datasets.make_regression(n_samples=5000, n_features=1, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

LR = LinearRegression()
LR.fit(X_train, y_train)
y1_test_predict = LR.predict(X_test)
Eout = np.mean( (y1_test_predict - y_test)**2 )
y_line = LR.predict(X)

LR2 = LinearRegression_GD(lr = 0.0001, n_itr=50000)
LR2.fit(X_train, y_train)
y2_test_predict = LR2.predict(X_test)
Eout2 = np.mean( (y2_test_predict - y_test)**2 )

fig = plt.figure(figsize=(8,6))
plt.plot(X, y_line, color='r')
plt.scatter(X_train[:,0], y_train, color='b' , marker='o', s=10)
plt.scatter(X_test[:,0], y_test, color='y', marker='o', s=20 )