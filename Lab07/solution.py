import util
import numpy
import scipy
import sklearn.datasets

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

def logreg_obj_wrapper(DTR , LTR , l):
    dim = DTR.shape[0]
    ZTR = LTR * 2.0 -1.0
    def logreg(v):
        w = util.vcol(v[0:dim])
        b = v[-1]
        scores = numpy.dot(w.T, DTR) + b #computes the score foreach training sample
        loss_per_sample = numpy.logaddexp(0 , -ZTR * scores)
        loss = loss_per_sample.mean() + 0.5 * l * numpy.linalg.norm(w)**2
        return loss
    return logreg

def train_logreg(D,L,lamb):
    logreg_obj = logreg_obj_wrapper(D,L , lamb)
    x0 = numpy.zeros(D.shape[0]+1)
    xOpt , fOpt , d = scipy.optimize.fmin_l_bfgs_b(logreg_obj,x0=x0,approx_grad=True)
    w , b  = util.vcol(xOpt[0:DTR.shape[0]]) , xOpt[-1]
    return w , b

if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    train_logreg(DTR, LTR, 1)