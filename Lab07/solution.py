import util
import numpy
import scipy

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
    x0 =
    xOpt , fOpt , d = scipy.optimize.fmin_l_bfgs_b(logreg_obj)
    w , b  = util.vcol(xOpt[0:DTR.shape[0]]) , xOpt[-1]
    return w , b

if __name__ == '__main__':
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)