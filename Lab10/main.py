import numpy
import math
import util
import scipy.special


def logpdf_GAU_ND(X, mu, C):
    _, log_determinant = numpy.linalg.slogdet(C)
    firstTerm = - (numpy.shape(X)[0] * 0.5) * math.log(2 * math.pi) - log_determinant * 0.5
    L = numpy.linalg.inv(C)
    XC = X - mu
    return firstTerm - 0.5 * (XC * numpy.dot(L,XC)).sum(0)

#SMV
def DUAL_LBFGS_KERNEL_SVM_NOBIA_Lambda(D,Y,C,pTar=None):
    n = float(D.shape[1])
    nT = float((Y>0).sum())
    nT = float((Y>0).sum())
def Kfunc_rbf_fact2(g):
    def k(D1,D2):
        DIST = util.vcol( (D1**2).sum(0) ) + util.vrow( (D2**2).sum(0) ) - 2* numpy.dor(D1.T , D2)
        return ;
    def K(D1,D2):
        H = numpy.zeros((D1.shape[1],D1.shape[2]))
        for i in range(D1.shape[1]):
            for j in range(D1.shape[2]) :
                H[i,j] = numpy.exp( -g * numpy.norm(x1-x2)**2)
def logpdf_GMM(X, gmm):
    (w, mu, C) = gmm
    yList = []
    for w , mu , C in gmm:
        lc = logpdf_GAU_ND(x,mu,X) + numpy.log()
        yList.append(lc)
    y = numpy.vstack(yList)
    return scipy.special.logsumexp(yList,axis=0)

def ML_GMM_IT(D,gmm):

    _ll = None
    deltaLL = 1.0