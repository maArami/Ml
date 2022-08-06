import numpy as np

# notice that if we has d feature our X matrix must has (N*(d+1)) demention (d+1 must be 1)  
class LinearRegression:
    def __init__(self):
        self.weights = None
        self.Error = None

    def fit(self, X, y):
        X_dager = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
        self.weights = np.dot(X_dager, y)
        self.Error = np.mean((np.dot(X, self.weights) - y)**2)

    def predict(self, X):
        return np.dot(X, self.weights)
