
import numpy as np
class Adaline(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state



    def fit(self, X, y):
        np.random.seed(16)
        #rgen = np.random.RandomState(self.random_state)
        self.w_ = np.random.uniform(-1, 1, size=1 + X.shape[1])
        self.cost_ = []
        self.error_=[]


        for i in range(self.n_iter):
            output = self.activation(X)
            errors = (y - output)
            self.error_.append(errors)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self.cost_

    def net_input(self, X):
        # calculate the dot product between y and x matrix
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):

        return self.net_input(X)

    def predict(self, X):

        return np.where(self.activation(X) >= 0.0, 1, -1)
