import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


def stepfunc(x):
    return np.where(x >= 0, 1, 0)


X, y = datasets.make_blobs(n_samples=500, n_features=2,
                           centers=2, cluster_std=4, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=120)


plt.scatter(x_train[:, 0], x_train[:, 1], marker='o', c=y_train)
plt.show()

N = 100  # number of itrate
w = np.zeros(shape=(len(X[1]), 1))
w_f = np.empty(shape=(len(X[1]), 1))
Ein = np.zeros(N)
w0 = 0
w0_f = 0

for k in range(N):
    for i in range(len(x_train)):
        y_p = stepfunc(np.dot(x_train[i], w)+w0)
        w = w + ((y_train[i] - y_p)*x_train[i:i+1]).T
        w0 = w0 + (y_train[i] - y_p)

    y_p = stepfunc(np.dot(x_train, w)+w0)
    Ein[k] = sum(abs(y_p[:, 0]-y_train))/len(y_train)

    if(k == 0):
        Ein_min = Ein[0]
        continue

    if(Ein[k] < Ein_min):
        w_f[:] = w
        w0_f = w0
        Ein_min = Ein[k]


x0_1 = np.amax(x_train[:, 0])
x0_2 = np.amin(x_train[:, 1])
x1_1 = -(w_f[0]*x0_1 + w0_f)/w_f[1]
x1_2 = -(w_f[0]*x0_2 + w0_f)/w_f[1]

plt.plot([x0_1, x0_2], [x1_1, x1_2])
plt.scatter(x_train[:, 0], x_train[:, 1], marker='o', c=y_train)

y_p = stepfunc(np.dot(x_train, w_f)+w0_f)
Ein_final = sum(abs(y_p[:, 0]-y_train))/len(y_train)  # final in-sample error

print(Ein_final)
