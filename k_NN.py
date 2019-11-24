import numpy as np

def k_NN(X, Y, x, k=1, p=2):
    """
    Input
    X: data [N, M]
    Y: label [N]
    x: forecast data [Q, M]
    k: number of nearest neighbor, default is 1
    p: Lp distance, default is 2

    Output
    y: forecast label [k]

    This algorithm can be implemented by kd-tree
    """
    if k > X.shape[0]:
        print("Error: k should be smaller than the number of data X")
    import heapq as hq
    y = []
    for x_i in x:
        que = [] # minimum heap
        label_y = None
        label_MAX = 0
        for i, X_i in enumerate(X):
            dis = np.sum(np.abs(x_i - X_i) ** (p)) ** (1 / p)
            hq.heappush(que, (dis, Y[i]))
        label_map = {}
        for i in range(k):
            if que[i][1] not in label_map.keys():
                label_map[que[i][1]] = 1
            else:
                label_map[que[i][1]] += 1
            if label_MAX < label_map[que[i][1]]:
                label_MAX = label_map[que[i][1]]
                label_y = que[i][1]
            hq.heappop(que)
        y.append(label_y)
    return y

X = np.random.rand(5, 5)
Y = np.array([1, 2, 3, 4, 5])
x = np.random.rand(10, 5)
print(k_NN(X, Y, x, 1))
