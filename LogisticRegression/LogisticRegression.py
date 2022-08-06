import numpy as np

class LogisticRegressin:
    def __init__(self, lr=0.0001, n_itr=1000):
        self.lr = lr
        self.n_itr = n_itr
        self.weightes = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weightes = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_itr):

            y_predict = self.predict(X)
            dw = (1 / n_samples)*np.dot(X.T, y_predict - y)
            db = (1 / n_samples)*np.sum(y_predict - y)

            self.weightes = self.weightes - self.lr*dw
            self.bias = self.bias - self.lr*db


    def predict(self, X):
        y_predict = self.sigmoid( np.dot(X, self.weightes) + self.bias )
        return y_predict
    
    def predict_cls(self, X):
        y_predict = self.predict(X)
        y_cls = np.array([1 if i>=0.5 else 0 for i in y_predict])
        return y_cls

    def sigmoid(self, X):
        return 1/(1+np.exp(-X))

    