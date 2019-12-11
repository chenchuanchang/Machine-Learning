import numpy as np

def PageRank(M, d=0.85):
    """
    Input
    M: transition matrix [N, N]
    d: damping factor, default is 0.85

    Output
    R: page rank vector[N]

    """
    I = np.eye(M.shape[0])
    d_I = np.ones((M.shape[0]))*((1-d)/M.shape[0])
    return np.matmul(np.linalg.inv(I-d*M),d_I)



# x = np.array([[0.,0., 1],[0.5,0.,0.],[0.5,1.,0]])
# print(PageRank(x))