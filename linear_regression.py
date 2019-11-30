import numpy as np

class linear_regression:
    def __init__(self, x, y, alg='ridge', phi=0.001):
        """
        Input
        x: data [N, M]
        y: value [N]
        alg: the algorithm of regression, default is 'ridge', {'lasso', 'ridge'}
        phi: learning rate

        Output
        w: weight vector [M+1]
        """
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        if alg == 'ridge':
            I = np.eye(x.shape[1], x.shape[1])
            self.w = np.matmul(np.matmul(y, x), np.linalg.pinv(np.matmul(x.T, x)+(phi/2)*I))
        else:
            cn = 0
            self.w = np.zeros((x.shape[1]))
            while True:
                cn += 1
                z = self.w - phi*(np.matmul(np.matmul(self.w, x.T), x)-np.matmul(y, x))
                w_n = np.array([x-phi if x>phi else x+phi for x in z])
                error = np.sum((w_n-self.w)**2)
                if error<0.000001:
                    break
                else:
                    self.w = w_n

    def fit(self, x):
        """
        Input
        x: forecast data [N, M]

        Output
        y: forecast value [N]
        """
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        y = [np.dot(self.w, x_i) for x_i in x]
        return y

# X = np.array([[i] for i in range(10)])
# Y = np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9,9.05])
# x = np.array([[i] for i in range(10)])
# model = linear_regression(X, Y, 'lasso')
# print(model.fit(x))