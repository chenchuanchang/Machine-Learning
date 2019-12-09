import numpy as np
import random
class k_means:
    def __init__(self, X, k):
        """
        Input
        X: data [N, M]
        k: number of clusters

        Output
        y: forecast label [N]

        """
        self.X = X
        self.k = k

    def dis(self, x, y):
        return np.sum((x-y)**2)

    def cluster(self):
        """

        Output
        y: forecast label [N]

        """
        init_x = self.X[[random.randint(0, self.X.shape[0]) for i in range(self.k)]]
        y = [-1 for i in range(self.X.shape[0])]
        while True:
            y_n = [np.argmin([self.dis(x_i, j) for j in init_x]) for x_i in self.X]
            flag = True
            for i in range(self.X.shape[0]):
                if y[i]!=y_n[i]:
                    flag = False
                    break
            if flag:
                return y_n
            x_k = [0 for i in range(self.k)]
            init_x = np.zeros((self.k, self.X.shape[1]))
            for i in range(self.X.shape[0]):
                init_x[y_n[i]] += self.X[i]
                x_k[y_n[i]] += 1
            init_x = np.array([np.zeros((self.X.shape[1])) if x_k[i] == 0 else init_x[i]/x_k[i] for i in range(self.k)])
            y = y_n

# X = np.random.rand(100, 5)
# model = k_means(X, 5)
# print(model.cluster())
