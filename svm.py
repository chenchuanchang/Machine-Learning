import numpy as np

class svm:
    def __init__(self, X, Y, C=10, kernel=None):
        """
        Input
        X: data [N, M]
        Y: label [N]
        C: parameter C
        kernel: kernel method, lambda function or default is K(x,y)=x*y

        """
        self.X = X
        self.Y = Y
        if kernel==None:
            self.kernel = lambda x, y: np.dot(x, y)
        else:
            self.kernel = kernel
        self.a = np.zeros((x.shape[0]))
        self.b = 0
        self.sv = []
        seta = 0.00001

        E = [-Y[i] for i in range(X.shape[0])]  # get Ei
        def Argmax(q):
            MAX = -np.inf
            inx = 0
            for i in range(len(E)):
                if i == q:
                    continue
                if E[i]>MAX:
                    MAX = E[i]
                    inx = i
            return inx
        def Argmin(q):
            MIN = np.inf
            inx = 0
            for i in range(len(E)):
                if i == q:
                    continue
                if E[i]<MIN:
                    MIN = E[i]
                    inx = i
            return inx
        while True:
            # print(self.a)
            a_1, a_2 = None, None # choose a_1, a_2
            g = []
            for i in range(X.shape[0]):
                g.append(self.b)
                for j in range(X.shape[0]):
                    g[i] = g[i] + Y[j] * self.a[j] * self.kernel(X[i], X[j])
            for i in range(X.shape[0]):
                E[i] = g[i]-Y[i]
            for i in range(X.shape[0]):
                if self.a[i]<C and self.a[i]>0 and abs(Y[i]*g[i]-1)>seta:
                    a_1 = i
                    if E[a_1]>0:
                        a_2 = Argmin(a_1)
                    else:
                        a_2 = Argmax(a_1)
            if a_1 == None:
                for i in range(X.shape[0]):
                    if abs(self.a[i])<seta and Y[i]*g[i]<1:
                        a_1 = i
                        if E[a_1] > 0:
                            a_2 = Argmin(a_1)
                        else:
                            a_2 = Argmax(a_1)
                    elif abs(self.a[i]-C)<seta and Y[i]*g[i]>1:
                        a_1 = i
                        if E[a_1] > 0:
                            a_2 = Argmin(a_1)
                        else:
                            a_2 = Argmax(a_1)
            if a_1 == None:
                for i in range(X.shape[0]):
                    if self.a[i] < C and self.a[i] > 0 and abs(Y[i] * g[i] - 1) < seta:
                        self.sv.append(i)
                break
            p = self.kernel(X[a_1],X[a_1])-2*self.kernel(X[a_1],x[a_2])+self.kernel(X[a_2],X[a_2])
            a_2_unc = self.a[a_2] + Y[a_2]*(E[a_1]-E[a_2])/p
            if Y[a_1]!=Y[a_2]:
                L = max(0, self.a[a_2]-self.a[a_1])
                H = min(C, self.a[a_2]-self.a[a_1]+C)
            else:
                L = max(0, self.a[a_2] + self.a[a_1]+C)
                H = min(C, self.a[a_2] + self.a[a_1])
            if a_2_unc > H:
                a_2_new = H
            elif a_2_unc<L:
                a_2_new = L
            else:
                a_2_new = a_2_unc
            a_1_new = self.a[a_1]+Y[a_1]*Y[a_2]*(self.a[a_2]-a_2_new)

            b_1_new = -E[a_1]-Y[a_1]*self.kernel(X[a_1],X[a_1])*(a_1_new-self.a[a_1])-Y[a_2]*self.kernel(X[a_2],X[a_1])*(a_2_new-self.a[a_2])+self.b
            b_2_new = -E[a_2]-Y[a_1]*self.kernel(X[a_1],X[a_2])*(a_1_new-self.a[a_1])-Y[a_2]*self.kernel(X[a_2],X[a_2])*(a_2_new-self.a[a_2])+self.b
            self.b = (b_1_new+b_2_new)/2
            self.a[a_1] = a_1_new
            self.a[a_2] = a_2_new
            print(self.a)

    def fit(self, x):
        """
        Input
        x: forecast data [N, M]

        Output
        y: forecast label [N]
        """
        y = []
        for i in range(x.shape[0]):
            y_v = self.b
            for s in self.sv:
                y_v = y_v + self.a[s]*self.Y[s]*self.kernel(x[i],self.X[s])
            y.append(1 if y_v>0 else -1)
        return np.array(y)

# x = np.array([[0.,0.],[1.,1.],[0.,1.],[1.,0.]])
# y = np.array([1,-1,1,-1])
# model = svm(x, y)
# print(model.fit(x))