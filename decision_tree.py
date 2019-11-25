import numpy as np
import copy
class node:
    def __init__(self, feature=None, sub=None, label=None):
        """
        Param
        feature: choose feature
        sub: feature set
        label: label for leaf node
        """
        self.feature = feature
        self.sub = sub
        self.label = label

def build_tree(X, Y, f_set, alg):
    p_y = {}  # p(y)
    node_y = None
    MAX = 0
    for i in range(Y.shape[0]):
        if Y[i] not in p_y.keys():
            p_y[Y[i]] = 1
        else:
            p_y[Y[i]] += 1
            if p_y[Y[i]] > MAX:
                MAX = p_y[Y[i]]
                node_y = Y[i]
    if len(f_set) == 0 or MAX == Y.shape[0]:
        return node(None, None, node_y)
    else:
        x_f = [{} for i in range(X.shape[1])]
        p_x_y = {}  # p(x|y)
        I_xy = {}  # I(x,y)
        H_y = 0
        for it in p_y.keys():
            C_k = p_y[it]/Y.shape[0]
            H_y = H_y - C_k*np.log2(C_k)

        inx = None
        MAX_H = 0
        for i in range(X.shape[0]):
            for j in f_set:
                if X[i][j] not in x_f[j].keys():
                    x_f[j][X[i][j]] = 1
                else:
                    x_f[j][X[i][j]] += 1
                if (X[i][j], Y[i], j) not in I_xy.keys():
                    I_xy[(X[i][j], Y[i], j)] = 1
                else:
                    I_xy[(X[i][j], Y[i], j)] += 1
        for j in f_set:
            H_x_y = 0
            H_a_D = 0
            for x in x_f[j].keys():
                p_di_d = x_f[j][x]/Y.shape[0]
                if alg == 'C45':
                    H_a_D = H_a_D - p_di_d*np.log2(p_di_d)
                for y in p_y.keys():
                    if (x, y, j) in I_xy.keys():
                        p_dik_di = I_xy[(x, y, j)]/x_f[j][x]
                        H_x_y = H_x_y - p_di_d * p_dik_di*np.log2(p_dik_di)
            H_x_y = (H_y-H_x_y)
            if alg == 'C45':
                H_x_y = H_x_y/H_a_D
            if H_x_y > MAX_H:
                MAX_H = H_x_y
                inx = j
        x_f_list = list(x_f[inx].keys())
        sub_x = [[] for i in range(len(x_f_list))]
        sub_y = [[] for i in range(len(x_f_list))]
        map_x_id = {}
        for i in range(len(x_f_list)):
            map_x_id[x_f_list[i]] = i
        for i, x_i in enumerate(X):
            sub_x[map_x_id[x_i[inx]]].append(x_i)
            sub_y[map_x_id[x_i[inx]]].append(Y[i])
        root_node = node(inx, {})
        new_f_set = copy.deepcopy(f_set)
        new_f_set.remove(inx)
        for i in range(len(x_f_list)):
            new_node = build_tree(np.array(sub_x[i]), np.array(sub_y[i]), new_f_set, alg)
            root_node.sub[x_f_list[i]] = new_node
        return root_node



def fit(dt, x):
    while dt.sub != None:
        dt = dt.sub[x[dt.feature]]
    return dt.label

def decision_tree(X, Y, x, alg = 'C45'):
    """
    Input
    X: data [N, M]
    Y: label [N]
    x: forecast data [Q, M]
    alg: the learning algorithm of decision tree , {'ID3', 'C45'}, default is 'C45'

    Output
    y: forecast label [k]

    Notice that each feature in X and x must be discrete
    """
    y = []
    f_set = [i for i in range(X.shape[1])]
    dt = build_tree(X, Y, f_set, alg) # build decision tree
    for x_i in x:
        y.append(fit(dt, x_i))
    return y



X = np.array([
    [1, 0, 0, '一般'],
    [1, 0, 0, '好'],
    [1, 1, 0, '好'],
    [1, 1, 1, '一般'],
    [1, 0, 0, '一般'],

    [2, 0, 0, '一般'],
    [2, 0, 0, '好'],
    [2, 1, 1, '好'],
    [2, 0, 1, '非常好'],
    [2, 0, 1, '非常好'],

    [3, 0, 1, '非常好'],
    [3, 0, 1, '好'],
    [3, 1, 0, '好'],
    [3, 1, 0, '非常好'],
    [3, 0, 0, '一般'],
])
Y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
x = np.array([
    [1, 0, 0, '一般'],
    [1, 0, 0, '好'],
    [1, 1, 0, '好'],
    [1, 1, 1, '一般'],
    [1, 0, 0, '一般'],

    [2, 0, 0, '一般'],
    [2, 0, 0, '好'],
    [2, 1, 1, '好'],
    [2, 0, 1, '非常好'],
    [2, 0, 1, '非常好'],

    [3, 0, 1, '非常好'],
    [3, 0, 1, '好'],
    [3, 1, 0, '好'],
    [3, 1, 0, '非常好'],
    [3, 0, 0, '一般'],
])
print(decision_tree(X, Y, x, 'ID3'))
# print(decision_tree(X, Y, x, 'C45'))