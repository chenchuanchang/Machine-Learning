import numpy as np

def perceptron(x, y, phi=0.1):
    """
    Input
    x: N feature [N, M]
    y: label [N]
    phi: learning rate

    Output
    w: weight vector [M]
    b: bias [1]

    Notice that perceptron can not be used for linear non separable data
    """
    w = np.zeros((x.shape[1]))
    b = 0
    while True:
        flag = True
        for i, x_i in enumerate(x):
            if y[i]*(w*x_i+b)<0:
                w = w - phi*y[i]*x_i
                b = b - phi*y[i]
                flag = False
        if flag:
            break
    return w, b