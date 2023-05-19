import numpy
import scipy
import sklearn.datasets
import util

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L


class logRegClass():

    def __init__(self, D, L, l):
        self.DTR = D
        self.ZTR = L * 2.0 - 1
        self.l = l
        self.dim = D.shape[0]
    def logreg_obj(self, v):
        # Compute and return the objective function value. You can retrieve all required information from self.DTR, self.LTR, self.l
        w = util.vcol(v[0: self.dim])
        b = v[-1]
        scores = numpy.dot(w.T, self.DTR) + b
        loss_per_sample = numpy.logaddexp(0 , -self.ZTR * scores)
        loss = loss_per_sample.mean()  + 0.5 * self.l + numpy.linalg.norm(w)**2
        return loss

    def train(self):
        x0 = numpy.zeros(self.DTR.shape[0]+1)
        xOpt , fOpt , d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj,x0=x0,approx_grad=True)
        w , b = util.vcol(xOpt[0:self.DTR.shape[0]]) , xOpt[-1]
        return w , b

def min_logregr_loss(DTE,LTE,w,b) :
    accuracy = 0
    LP = []
    for i in range(DTE.shape[1]):
        score = numpy.dot(util.vrow(w), util.vcol(DTE[:, i])) + b
        if score > 0:
            LP.append(1)
        else:
            LP.append(0)

        if LP[i] == LTE[i]:
            accuracy += 1

    print("Accuracy")
    print(accuracy / DTE.shape[1])
    print(" Error ")
    print((DTE.shape[1] - accuracy) / DTE.shape[1])

if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    for lam in [1 , 0.1 , 1e-3 , 1e-6]:
        logRegObj = logRegClass(DTR, LTR, lam)
        w , b = logRegObj.train()
        print(w , b)
        #min_logregr_loss(DTE, LTE, w , b)