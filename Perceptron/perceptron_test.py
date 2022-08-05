from sklearn.model_selection import train_test_split
from sklearn import datasets

from Perceptron import Perseptron
X, y = datasets.make_blobs(n_samples=500, n_features=3,
                           centers=2, cluster_std=3, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)


# plt.scatter(x_train[:, 0], x_train[:, 1], marker='o', c=y_train)
# plt.show()

p = Perseptron(500)
p.fit(x_train, y_train)

y_predicted = p.predict(x_test)
Eout = sum(abs(y_predicted[:, 0] - y_test))/len(y_test)
print(p.Error)