import numpy as np
import copy
class node:
    def __init__(self, feature=None, threshold=None, sub=None, label=None):
        """
        Param
        feature: choose feature
        sub: feature set
        label: label for leaf node
        """
        self.feature = feature
        self.threshold = threshold
        self.sub = sub
        self.label = label

class CART:
    def __init__(self, X, Y, max_depth=15):
        """
        Input
        X: data [N, M]
        Y: value [N]
        max_depth: the max depth of regression decision tree, default is 15

        Least squares regression tree
        """
        self.dt = self.build_tree(X, Y, 0, max_depth)

    def build_tree(self, X, Y, now_depth, max_depth):
        if now_depth >= max_depth:
            return node(None, None, None, np.sum(Y)/Y.shape[0])
        point = None
        loss_min = -1
        flag = False
        for j in range(X.shape[1]):
            for s in range(X.shape[0]):
                x_1_n, x_2_n = 0, 0
                y_1_sum2, y_2_sum2 = 0, 0
                y_1_sum1, y_2_sum1 = 0, 0
                for x_i in X:
                    if x_i[j] < X[s][j]:
                        y_1_sum1 += Y[s]
                        y_1_sum2 += (Y[s]**2)
                        x_1_n += 1
                    else:
                        y_2_sum1 += Y[s]
                        y_2_sum2 += (Y[s] ** 2)
                        x_2_n += 1
                if x_1_n == 0:
                    loss_1 = 0
                else:
                    c_1 = y_1_sum1/x_1_n
                    loss_1 = y_1_sum2 - 2 * c_1 * y_1_sum1 + x_1_n * (c_1 ** 2)
                if x_2_n == 0:
                    loss_2 = 0
                else:
                    c_2 = y_1_sum1 / x_2_n
                    loss_2 = y_2_sum2 - 2 * c_2 * y_2_sum1 + x_2_n * (c_2 ** 2)
                if loss_min == -1 or loss_min>(loss_1+loss_2):
                    if x_1_n == X.shape[0] or x_2_n == X.shape[0]:
                        flag = True
                    else:
                        flag = False
                    loss_min = loss_1+loss_2
                    point = [j, X[s][j]]
        if flag:
            return node(None, None, None, np.sum(Y) / Y.shape[0])
        X_1, X_2, Y_1, Y_2 = [], [], [], []
        for i in range(X.shape[0]):
            if X[i][point[0]]<point[1]:
                X_1.append(X[i])
                Y_1.append(Y[i])
            else:
                X_2.append(X[i])
                Y_2.append(Y[i])
        new_node = node(point[0], point[1], [], None)
        new_node.sub.append(self.build_tree(np.array(X_1), np.array(Y_1), now_depth+1, max_depth))
        new_node.sub.append(self.build_tree(np.array(X_2), np.array(Y_2), now_depth + 1, max_depth))
        return new_node
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
            t_dt = self.dt
            while t_dt.sub != None:
                if x_i[t_dt.feature] < t_dt.threshold:
                    t_dt = t_dt.sub[0]
                else:
                    t_dt = t_dt.sub[1]
            y.append(t_dt.label)
        return y

X = np.array([[i] for i in range(10)])
Y = np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9,9.05])
x = np.array([[i] for i in range(10)])
model = CART(X, Y)
print(model.fit(x))