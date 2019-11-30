import numpy as np
import random
from CART import CART

class AdaBoost:  # 2-classification for example
    def __init__(self, x, y, model='CART', error=0.01):
        """
        Input
        x: data [N, M]
        y: label [N]
        model: base model, default is 'CART'
        error: training error

        """
        self.M = 0
        self.a = []
        self.G =[]
        D = np.array([1/x.shape[0] for i in range(x.shape[0])])
        E = 100000000000000
        while E > error:
            if model == 'CART':
                self.G.append(CART(x,y, 'classification', max_depth=2, w=D))
            y_ = np.sign(self.G[self.M].fit(x))
            em = np.sum([D[i]*(0 if y_[i]==y[i] else 1) for i in range(x.shape[0])])
            if em<0.00001:
                self.a.append(1)
                self.M += 1
                break
            self.a.append(0.5*np.log((1-em)/em))
            Z = np.sum([D[i]*np.exp(-self.a[self.M]*y[i]*y_[i]) for i in range(x.shape[0])])
            D = D/Z
            self.M += 1
            y_t = self.fit(x)
            E = np.sum([(1 if y_t[i]==y[i] else 0) for i in range(x.shape[0])])/x.shape[0]

    def fit(self, x):
        """
        Input
        x: forecast data [N, M]

        Output
        y: forecast label [N]
        """
        y = np.zeros((x.shape[0]))
        for m in range(self.M):
            y = y + self.a[m]*self.G[m].fit(x)
        return np.sign(y)

# X = np.array([[0., 0.], [1., 1.], [1, -1]])
# Y = np.array([1, -1, 1])
# model = AdaBoost(X, Y)
# print(model.fit(X))