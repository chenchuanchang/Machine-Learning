import numpy as np
import random

class HMM_BW:
    def __init__(self, O, O_num, I_num):
        """
        Input
        O: observe sequences [S, T]
        O_num: type number of observe
        I_num: type number of state

        Param
        A: estimate state transfer matrix [I_num, I_num]
        B: estimate observe transfer matrix [O_num, O_num] 
        pi: estimate initial state probability [I_num]
        """
        pass

class HMM_VB:
    def __init__(self, A, B, pi):
        """
        Input
        A: state transfer matrix [I_num, I_num]
        B: observe transfer matrix [O_num, O_num]
        pi: initial state probability [I_num]

        """
        self.A = A
        self.B = B
        self.pi = pi

    def predict(self, O):
        """
        Input
        O: observe sequences [S, T]

        Output
        I: forecast state sequences [S, T]
        """
        pass
# O = np.array([[0,1,1,2],[0,1,2,1],[0,1,1,1]])
# model = HMM_BW(O,3,3)
# print(model.A,model.B,model.pi)