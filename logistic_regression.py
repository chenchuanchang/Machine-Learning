import numpy as np
import random
class logistic_regression:
    def __init__(self, x, y, phi=0.1, l=0.01, epoch=1000):
        """
        Input
        x: data [N, M]
        y: label [N]
        phi: learning rate
        l: parameter of regularization term
        epoch: iteration

        Output
        w: weight vector [M+1]

        This is softmax regression
        """
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        set_y = set(y)
        self.k = len(set_y)
        self.y_ma = {_y:i for i, _y in enumerate(set_y)}
        self.ma_y = {i:_y for i, _y in enumerate(set_y)}
        self.w = np.zeros((len(set_y), x.shape[1]))
        ran_x = [i for i in range(x.shape[0])]
        cn = 0
        while cn<epoch:
            id = random.sample(ran_x, 1)[0]
            e_w_x = np.zeros((self.k))
            for j in range(self.k):
                e_w_x[j] = np.exp(np.dot(self.w[j], x[id]))
            e_w_x_a = np.sum(e_w_x)
            e_w_x /= e_w_x_a
            j = self.y_ma[y[id]]
            seta = x[id]*(e_w_x[j]-1)+l*self.w[j]
            self.w[j] = self.w[j] - phi*seta
            cn += 1

    def fit(self, x):
        """
        Input
        x: forecast data [N, M]

        Output
        y: forecast label [N]
        """
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        y = []
        for x_i in x:
            e_w_x = np.zeros((self.k))
            for j in range(self.k):
                e_w_x[j] = np.exp(np.dot(self.w[j], x_i))
            e_w_x_a = np.sum(e_w_x)
            e_w_x /= e_w_x_a
            y.append(self.ma_y[np.argmax(e_w_x)])
        return y

# x = np.array([[0.,0.],[1.,1.], [1,-1]])
# y = np.array([1,-1,2])
# model = logistic_regression(x, y)
# print(model.fit(x))