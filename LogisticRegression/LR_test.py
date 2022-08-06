
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from LogisticRegression import LogisticRegression


bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

lr = LogisticRegression(lr=0.0001, n_itr=1000)
lr.fit(X_train, y_train)

y_p = lr.predict_cls(X_test)

accurancy = np.mean(y_p == y_test)
