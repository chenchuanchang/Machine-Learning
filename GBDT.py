import numpy as np
import random
from CART import CART

class GBDT:  # 1-regression for example
    def __init__(self, x, y, model='CART', error=0.001):
        """
        Input
        x: data [N, M]
        y: label [N]
        model: base model, default is 'CART'
        error: training error

        """
        self.M = 0
        self.G =[]
        E = 100000000000000
        y_m = y
        while E > error:
            if model == 'CART':
                self.G.append(CART(x, y_m, 'regression', max_depth=1))
            self.M += 1
            y_t = self.fit(x)
            E = np.sum((y_t-y)**2)
            y_m = y-y_t

    def fit(self, x):
        """
        Input
        x: forecast data [N, M]

        Output
        y: forecast label [N]
        """
        y = np.zeros((x.shape[0]))
        for m in range(self.M):
            y = y + self.G[m].fit(x)
        return y

# X = np.array([[i+1] for i in range(10)])
# Y = np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9,9.05])
# model = GBDT(X, Y)
# print(model.fit(X))