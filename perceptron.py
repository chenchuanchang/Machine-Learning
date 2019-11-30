import numpy as np

class perceptron:
    def __init__(self, x, y, phi=0.1):
        """
        Input
        x: data [N, M]
        y: label [N]
        phi: learning rate

        Notice that perceptron can not be used for linear non separable data
        """
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        self.w = np.zeros((x.shape[1]))
        while True:
            flag = True
            for i, x_i in enumerate(x):
                if y[i] * np.dot(self.w, x_i) <= 0:
                    self.w = self.w + phi * y[i] * x_i
                    flag = False
                    break
            if flag:
                break
    def fit(self, x):
        """
        Input
        x: forecast data [N, M]

        Output
        y: forecast label [N]
        """
        x = np.hstack((x, np.ones((x.shape[0], 1))))
        y = [1 if np.dot(self.w, x_i)>0 else -1 for x_i in x]
        return y

# x = np.array([[0.,0.],[1.,1.]])
# y = np.array([1,-1])
# model = perceptron(x, y)
# print(model.fit(x))