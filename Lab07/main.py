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
    def __init__(self, DTR, LTR, l=1e-3):
        self.DTR = DTR
        self.ZTR = LTR * 2.0 - 1
        self.l = l
        self.dim = DTR.shape[0]

    def logreg_obj(self, v):
        # Compute and return the objective function value. You can retrieve all required information from self.DTR, self.LTR, self.l
        w = util.vcol(v[0: self.dim])
        b = v[-1]
        scores = numpy.dot(w.T, self.DTR) + b
        loss_per_sample = numpy.logaddexp(0 , -self.ZTR * scores)
        loss = 0.5 * self.l * numpy.linalg.norm(w)**2 + loss_per_sample.mean()
        return loss

    def train(self):
        x0 = numpy.zeros(self.DTR.shape[0]+1)
        xOpt , fOpt , d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj,x0=x0,approx_grad=True)
        w , b = util.vcol(xOpt[0:self.DTR.shape[0]]) , xOpt[-1]
        return w , b

    def evaluate(self,DTE,LTE):
        w, b = self.train()
        Score = numpy.dot(w.T, DTE) + b
        PLabel = (Score > 0).astype(int)
        Error = ((LTE != PLabel).astype(int).sum() / DTE.shape[1]) * 100
        return Error , w, b

def BinaryLogRegression(DTR, LTR, DTE, LTE):
    minRate = 100
    w_best, b_best = (0, 0)
    for lam in [1, 0.1, 1e-3, 1e-6]:
        Error, w, b = logRegClass(DTR, LTR, lam).evaluate(DTE, LTE)
        print(Error)
        if Error < minRate:
            minRate = Error
            w_best, b_best = w, b
    return minRate , w_best, b_best

if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    _ , w , b = BinaryLogRegression(DTR, LTR, DTE, LTE)
