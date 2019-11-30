import numpy as np

class naive_Bayes:
    def __init__(self, X, Y, L=0):
        """
        Input
        X: data [N, M]
        Y: label [N]
        L: Laplacian operator, default is 0

        Notice that each feature in X must be discrete
        """
        self.x_f = [set() for i in range(X.shape[1])]
        self.p_y = {}  # p(y)
        self.p_x_y = {}  # p(x|y)
        self.I_xy = {}  # I(x, y)
        for i in range(Y.shape[0]):
            if Y[i] not in self.p_y.keys():
                self.p_y[Y[i]] = 1
            else:
                self.p_y[Y[i]] += 1
        k = len(self.p_y.keys())
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i][j] not in self.x_f[j]:
                    self.x_f[j].add(X[i][j])
                if (X[i][j], Y[i], j) not in self.I_xy.keys():
                    self.I_xy[(X[i][j], Y[i], j)] = 1
                else:
                    self.I_xy[(X[i][j], Y[i], j)] += 1
        for it in self.I_xy.keys():
            self.p_x_y[it] = (self.I_xy[it] + L) / (self.p_y[it[1]] + len(self.x_f[it[2]]))
        for it in self.p_y.keys():
            self.p_y[it] = (self.p_y[it] + L) / (X.shape[0] + k * L)

    def fit(self, x):
        """
        Input
        x: forecast data [Q, M]

        Output
        y: forecast label [k]

        Notice that each feature in x must be discrete
        """
        y = []
        for x_i in x:
            MAX = 0
            y_tar = None
            for y_i in self.p_y.keys():
                p_y_x = self.p_y[y_i]
                for j in range(X.shape[1]):
                    p_y_x = p_y_x * self.p_x_y[(x_i[j], y_i, j)]
                if p_y_x > MAX:
                    MAX = p_y_x
                    y_tar = y_i
            y.append(y_tar)
        return y

# X = np.array([
#     [1, 'S'],
#     [1, 'M'],
#     [1, 'M'],
#     [1, 'S'],
#     [1, 'S'],
#
#     [2, 'S'],
#     [2, 'M'],
#     [2, 'M'],
#     [2, 'L'],
#     [2, 'L'],
#
#     [3, 'L'],
#     [3, 'M'],
#     [3, 'M'],
#     [3, 'L'],
#     [3, 'L'],
# ])
# Y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
# x = np.array([
#     [2, 'S']
# ])
# model = naive_Bayes(X, Y)
# print(model.fit(x))