import numpy as np

# notice that if we has d feature our X matrix must has (N*(d+1)) demention (d+1 must be 1)  
class LinearRegression:
    def __init__(self):
        self.weights = None
        self.Error = None

    def fit(self, _X, y):
        n_samples, n_features = _X.shape
        X = np.ones((n_samples,n_features+1))
        X[:,0:n_features] = _X

        X_dager = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        self.weights = np.dot(X_dager, y)
        self.Error = np.mean((np.dot(X, self.weights) - y)**2)

    def predict(self, _X):
        n_samples, n_features = _X.shape
        X = np.ones((n_samples,n_features+1))
        X[:,0:n_features] = _X

        return np.dot(X, self.weights)
#----------------------------------------------------------------------------

#linear Regresion with GD method
class LinearRegression_GD:

    def __init__(self, lr=0.001, n_itr=10000):
        self.lr = lr
        self.n_itr= n_itr
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_itr):
            y_predict = self.predict(X)
            dw = (2/n_samples)*np.dot(X.T, (y_predict - y))
            db = (2/n_samples)*np.sum( y_predict - y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        y_predict = np.dot(X, self.weights) + self.bias
        return y_predict