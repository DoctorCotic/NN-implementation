import numpy as np
import pandas as pd

SAMPLE_DATA = 'data.csv'

def main():
    fd = pd.read_csv(SAMPLE_DATA, sep=',')
    for line in fd.values:
        print(line[0], line[4])


if __name__ == '__main__':
    main()

class Perceptron(object):
    """ Классификатор на основе персептрона
    """
    def __init__(self, eta=0.01, n_iter = 10):
        self.eta = eta
        self.n_iter=n_iter
    #выполняет подгонку модели
    def fit(self, X, y):
        self.w_= np.zeros(1+ X.Shape[1])
        self.errors_=[]

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0]  += update
                errors      += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self,X ):
        """clear enter"""
        return np.dot(X, self.w_[1:])+self.w_[0]
    #выполнение прогнозов
    def predict(self, X):
        """return class label"""
        return np.where(self.net_input(X)>= 0.0, 1, -1)
