import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.001, n_itr=500):
        self.lr = lr
        self.n_itr = n_itr
        self.weights = None
        self.Error = None

    def fit(self, _X, y):
        n_samples, n_features = _X.shape
        X = np.ones((n_samples, n_features+1)) #insert [1] column in matrix
        X[:,0:n_features] = _X 
        y = np.array([-1 if i==0 else +1 for i in y])

        self.weights = np.zeros(n_features+1)
        Egrad = np.zeros(n_features+1)
        
        for _ in range(self.n_itr):

            for i in range(n_samples):
                Egrad = Egrad + (y[i]*X[i,:])/(1 + np.exp(y[i]*np.dot(self.weights,X[i,:])))
            Egrad /= (-n_samples)
            self.weights = self.weights - self.lr*Egrad
            Egrad *= 0
        

    def predict(self, _X):
        n_samples, n_features = _X.shape
        X = np.ones((n_samples, n_features+1)) #insert [1] column in matrix
        X[:,0:n_features] = _X 

        return self.sigmoid(np.dot(X,self.weights))
    
    def predict_cls(self, X):
        y_predict = self.predict(X)
        y_predict = np.array([+1 if i>=0.5 else 0 for i in y_predict])
        return y_predict 

    def sigmoid(self, X):
        return 1/( 1 + np.exp(-X))