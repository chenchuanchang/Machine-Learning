import numpy as np

def naive_Bayes(X, Y, x, L=0):
    """
    Input
    X: data [N, M]
    Y: label [N]
    x: forecast data [Q, M]
    L: Laplacian operator, default is 0

    Output
    y: forecast label [k]

    Notice that each feature in X and x must be discrete
    """
    x_f = [set() for i in range(X.shape[1])]
    p_y = {}
    p_x_y = {}
    I_xy = {}
    y = []
    for i in range(Y.shape[0]):
        if Y[i] not in p_y.keys():
            p_y[Y[i]] = 1
        else:
            p_y[Y[i]] += 1
    k = len(p_y.keys())
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[1][j] not in x_f[j]:
                x_f[i].add(X[i][j])
            if (X[i][j], Y[i], j) not in I_xy.keys():
                I_xy[(X[i][j], Y[i], j)] = 1
            else:
                I_xy[(X[i][j], Y[i], j)] += 1
    for it in I_xy.keys():
        p_x_y[it] = (I_xy[it]+L)/(p_y[it[1]]+len(x_f[it[2]]))
    for it in p_y.keys():
        p_y[it] = (p_y[it]+L)/(X.shape[0]+k*L)
    for x_i in x:
        MAX = 0
        y_tar = None
        for y_i in p_y.keys():
            p_y_x = p_y[y_i]
            for j in range(X.shape[1]):
                p_y_x = p_y_x*p_x_y[(x_i[j], y_i, j)]
            if p_y_x>MAX:
                MAX = p_y_x
                y_tar = y_i
        y.append(y_tar)
    return y



X = np.array([
    [1, 'S'],
    [1, 'M'],
    [1, 'M'],
    [1, 'S'],
    [1, 'S'],

    [2, 'S'],
    [2, 'M'],
    [2, 'M'],
    [2, 'L'],
    [2, 'L'],

    [3, 'L'],
    [3, 'M'],
    [3, 'M'],
    [3, 'L'],
    [3, 'L'],
])
Y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
x = np.array([
    [2, 'S']
])
print(naive_Bayes(X, Y, x))