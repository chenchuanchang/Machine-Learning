import numpy as np

def perceptron(x, y, phi=0.1):
    """
    Input
    x: data [N, M]
    y: label [N]
    phi: learning rate

    Output
    w: weight vector [M]
    b: bias [1]

    Notice that perceptron can not be used for linear non separable data
    """
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    w = np.zeros((x.shape[1]))
    while True:
        flag = True
        for i, x_i in enumerate(x):
            if y[i]*np.dot(w, x_i)<=0:
                w = w + phi*y[i]*x_i
                flag = False
                break
        if flag:
            break
    return w[:-1], w[-1]

x = np.array([[0.,0.],[1.,1.]])
y = np.array([1,-1])
print(perceptron(x, y))