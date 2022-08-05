import numpy as np


class Perseptron:
    def __init__(self, N=100):
        self.N = N
        self.w = None
        self.w0 = None
        self.activation_func = self.step_func
        self.W = None
        self.W0 = None
        self.Error = None

    def fit(self, X, y):  # this method return the Wieghted of perceptron with pocket algorithm
        n_samples, n_features = X.shape

        self.W = np.zeros((n_features, 1))
        self.W0 = 0

        W_tmp = np.zeros((n_features, 1))
        W0_tmp = 0

        for k in range(self.N):
            for i in range(n_samples):
                y_predicted = self.step_func(np.dot(X[i], W_tmp) + W0_tmp)
                W_tmp = W_tmp + ((y[i] - y_predicted)*X[i:i+1]).T
                W0_tmp = W0_tmp + (y[i] - y_predicted)

            y_predicted = self.step_func(np.dot(X, W_tmp) + W0_tmp)
            Ein = sum(abs(y_predicted[:, 0]-y))/len(y)

            if(k == 0):
                self.W = W_tmp
                self.W0 = W0_tmp
                Ein_min = Ein
                continue

            if(Ein < Ein_min):
                self.W[:] = W_tmp
                self.W0 = W0_tmp
                Ein_min = Ein

        self.Error = Ein_min

    def predict(self, X):
        y_predicted = self.activation_func(np.dot(X, self.W) + self.W0)
        return y_predicted

    def step_func(self, x):
        return np.where(x >= 0, 1, 0)
