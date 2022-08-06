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
