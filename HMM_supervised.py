import numpy as np
import random

class HMM:
    def __init__(self, O, I, O_num, I_num):
        """
        Input
        O: observe sequences [S, T]
        I: state sequences [S, T]
        O_num: type number of observe
        I_num: type number of state

        Param
        A: state transfer matrix [I_num, I_num]
        B: observe transfer matrix [O_num, O_num]
        pi: initial state probability [I_num]
        """
        A_ = np.zeros((I_num, I_num))
        B_ = np.zeros((I_num, O_num))
        pi_ = np.zeros((I_num))
        for s in range(O.shape[0]):
            pi_[O[s][0]] += 1
            for t in range(O.shape[1]-1):
                A_[O[s][t]][O[s][t+1]]+=1
                B_[I[s][t]][O[s][t]]+=1
            B_[I[s][-1]][O[s][-1]] += 1
        A_sum = np.squeeze(np.sum(A_, axis=-1))
        B_sum = np.squeeze(np.sum(B_, axis=-1))
        self.A = np.array([np.zeros((I_num)) if A_sum[i] == 0 else A_[i]/A_sum[i] for i in range(I_num)])
        self.B = np.array([np.zeros((O_num)) if B_sum[i] == 0 else B_[i]/B_sum[i] for i in range(I_num)])
        self.pi = pi_/O.shape[0]

# O = np.array([[0,0,0,1],[0,0,1,0],[0,0,0,0]])
# I = np.array([[0,1,1,2],[0,1,2,1],[0,1,1,1]])
# model = HMM(O,I,2,3)
# print(model.A,model.B,model.pi)