import numpy as np
import random

class SVD:
    def __init__(self, A, alg='truncated', k=None):
        """
        Input
        A: matrix [N, M]
        alg: decomposition type, {'compact', 'truncated'} default if 'truncated'

        Param
        U: left matrix [N, N]
        S: singular value matrix {[r, r], [k, k]} depended on alg
        V: right matrix [M, M]
        """

        s, v = np.linalg.eig(np.dot(A.T, A))
        v = v.T
        s_ = [[s[i], i] for i in range(A.shape[1])]
        s_ = sorted(s_, key=lambda x:x[0], reverse=True)
        r = 0
        while r<s.shape[0] and s_[r][0] != 0:
            r += 1
        s_ = s_[:r]
        s = np.array([s_[i][0] for i in range(len(s_))])
        self.V = np.array([v[s_[i][1]] for i in range(s.shape[0])])
        if alg == 'compact':
            self.S = np.diag(s)
            self.V = self.V[:s.shape[0]].T
            self.U = np.dot(np.dot(A, self.V), np.linalg.inv(self.S))
        else:
            if k>=r:
                k = r
            self.S = np.diag(s[:k])
            self.V = self.V[:k].T
            self.U = np.dot(np.dot(A, self.V), np.linalg.inv(self.S))


# A = np.array([[1,0,0,0],[0,0,0,4],[0,3,0,0],[2,0,0,0]])
# model = SVD(A, alg = 'compact')
# print(model.U)
# print(model.S)
# print(model.V)
# print(np.dot(np.dot(model.U, model.S), model.V.T))