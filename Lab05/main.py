import numpy
import scipy.special
import sklearn.datasets
import util

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def load_digits():
    D, L = sklearn.datasets.load_digits()['data'].T, sklearn.datasets.load_digits()['target']
    #D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]    #Data Training
    DTE = D[:, idxTest]     #Data Test
    LTR = L[idxTrain]       #Label Training
    LTE = L[idxTest]        #Label Test
    return (DTR, LTR), (DTE, LTE)

def logpdf_GAU_ND(X, mu, C):
    _, log_determinant = numpy.linalg.slogdet(C)
    firstTerm = - (numpy.shape(X)[0] * 0.5) * numpy.log(2 * numpy.pi) - log_determinant * 0.5
    L = numpy.linalg.inv(C)
    XC = X - mu
    return firstTerm - 0.5 * (XC * numpy.dot(L,XC)).sum(0)

#MultivariateGaussianClassifier
def linearMVG(DTR, LTR, DTE):
    hCls = {}
    for lab in [0, 1, 2]:
        DCLS = DTR[:, LTR == lab]
        hCls[lab] = util.dataCovarianceMatrix(DCLS)
    ### Classification
    prior = util.vcol(numpy.ones(3) / 3)
    S = []
    for hyp in [0, 1, 2]:
        C , mu = hCls[hyp]
        fcond = numpy.exp(logpdf_GAU_ND(DTE, mu, C))
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S * prior
    P = SJoint / util.vrow(SJoint.sum(0)) # vrow(S.sum(0)) is the marginal

    return SJoint, P

def logMVG(DTR, LTR, DTE):
    hCls = {}
    for lab in [0, 1, 2]:
        DCLS = DTR[:, LTR == lab]
        hCls[lab] = util.dataCovarianceMatrix(DCLS)
    #classfication
    logprior = numpy.log(util.vcol(numpy.ones(3) / 3))
    S = []
    for hyp in [0, 1, 2]:
        C , mu = hCls[hyp]
        fcond = logpdf_GAU_ND(DTE, mu, C)
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S + logprior   #S is the logJoint
    logP = SJoint - util.vrow(scipy.special.logsumexp(SJoint,0))
    #logsumexp does in a numerical stable way the sum and the exp for marginal
    P = numpy.exp(logP)
    return SJoint , P

def logNaiveMVG(DTR, LTR, DTE):
    hCls = {}
    for lab in [0, 1, 2]:
        DCLS = DTR[:, LTR == lab]
        C , mu = util.dataCovarianceMatrix(DCLS)
        ones = numpy.diag(numpy.ones(DCLS.shape[0]))
        hCls[lab] = (C * ones,mu)

    logprior = numpy.log(util.vcol(numpy.ones(3) / 3))
    S = []
    for hyp in [0, 1, 2]:
        C, mu = hCls[hyp]
        fcond = logpdf_GAU_ND(DTE, mu, C)
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S + logprior  # S is the logJoint
    logP = SJoint - util.vrow(scipy.special.logsumexp(SJoint, 0))
    # logsumexp does in a numerical stable way the sum and the exp for marginal
    P = numpy.exp(logP)

    return SJoint , P

def logTiedMVG(DTR, LTR, DTE):
    hCls = {}
    Sw = numpy.zeros((4, 4))
    for i in range(0, 3):
        Sw += util.within_class_covariance(DTR[:, LTR == i], DTR.size)

    for lab in [0, 1, 2]:
        DCLS = DTR[:, LTR == lab]
        _ , mu = util.dataCovarianceMatrix(DCLS)
        hCls[lab] = (Sw,mu)

    logprior = numpy.log(util.vcol(numpy.ones(3) / 3))
    S = []
    for hyp in [0, 1, 2]:
        C, mu = hCls[hyp]
        fcond = logpdf_GAU_ND(DTE, mu, C)
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S + logprior  # S is the logJoint
    logP = SJoint - util.vrow(scipy.special.logsumexp(SJoint, 0))
    # logsumexp does in a numerical stable way the sum and the exp for marginal
    P = numpy.exp(logP)

    return SJoint, P

def Classificators():
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    _, P = linearMVG(DTR, LTR, DTE)
    # test = numpy.load('solution/SJoint_MVG.npy')
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100
    print(error)

    _, P = logMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100
    print(error)

    _, P = logNaiveMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100
    print(error)

    _, P = logTiedMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100
    print(error)

def split_in_k(D,L,k, seed=0):
    #we want to mantain the K (#of models / groups) as large as possible to emulate the "leave-one-out"
    #for IRIS is ok to implement the leave-one-out
    #D, L = load_iris()

    k_folds = []
    label_k_folds = []
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxSplit = numpy.array_split(idx, k)
    for i in range(k):
        k_folds.append(D[:, idxSplit[i]])
        label_k_folds.append(L[idxSplit[i]])
    return k_folds , label_k_folds

if __name__ == '__main__':
    #Classificators()
    D, L = load_iris()
    n = 1
    k = D.shape[1]//n
    print(f" k = {k} and n = {n}")
    errors_MVG = 0
    errors_logMVG = 0
    errors_Naive = 0
    errors_Tied = 0
    groups , groups_labels = split_in_k(D,L,k)
    for i in range(k):
        temp = groups[:i] + groups[i+1:]
        DTR = numpy.hstack(temp)
        temp = groups_labels[:i] + groups_labels[i+1:]
        LTR = numpy.hstack(temp)
        DTE = groups[i]
        LTE = groups_labels[i]

        _, P = linearMVG(DTR, LTR, DTE)
        # test = numpy.load('solution/SJoint_MVG.npy')

        SPost = P.argmax(axis=0)
        errors_MVG += numpy.abs ((SPost - LTE).sum())

        _, P = logMVG(DTR, LTR, DTE)
        SPost = P.argmax(axis=0)
        errors_logMVG += numpy.abs ((SPost - LTE).sum())

        _, P = logNaiveMVG(DTR, LTR, DTE)
        SPost = P.argmax(axis=0)
        errors_Naive += numpy.abs ((SPost - LTE).sum())

        _, P = logTiedMVG(DTR, LTR, DTE)
        SPost = P.argmax(axis=0)
        errors_Tied += numpy.abs ((SPost - LTE).sum())

    print(f"Errors MVG : {errors_MVG/150*100} %")
    print(f"Errors logMVG : {errors_logMVG / 150 * 100} %")
    print(f"Errors Naive : {errors_Naive / 150 * 100} %")
    print(f"Errors Tied {errors_Tied / 150 * 100}%")