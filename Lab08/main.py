import numpy
import scipy.special
import sklearn.datasets
import util
import matplotlib.pyplot as plt

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

def confusion_matrix(LTE,SPost):
    n = numpy.unique(LTE).shape[0]
    matrix = numpy.zeros([n,n])
    for i in numpy.unique(LTE):
        for j in numpy.unique(LTE):
            matrix[i][j] = (SPost[LTE == j] == i).sum()
    return matrix

def Classificator_iris():
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    _, P = linearMVG(DTR, LTR, DTE)
    # test = numpy.load('solution/SJoint_MVG.npy')
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100

    confusion_matrix(LTE, SPost)

    _, P = logMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100

    confusion_matrix(LTE, SPost)

    _, P = logNaiveMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100

    confusion_matrix(LTE,SPost)

    _, P = logTiedMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100

    confusion_matrix(LTE,SPost)

def Discriminant_ratio(threshold,SPost):
    res = numpy.copy(SPost)
    res[SPost > threshold] = 1
    res[SPost <= threshold] = 0
    return res

def Compute_Anormalized_DCF(matrix, pi, C_fn, C_fp):
    FNR = matrix[0][1] / (matrix[0][1] + matrix[1][1])
    FPR = matrix[1][0] / (matrix[0][0] + matrix[1][0])

    DCF = pi * C_fn * FNR + (1-pi) * C_fp * FPR
    return DCF

def Compute_Normalized_DCF(DCF,pi,C_fn,C_fp):
    optimal_risk = numpy.min([pi * C_fn, (1 - pi) * C_fp])
    return DCF / optimal_risk

def Compute_DCF(matrix, pi, C_fn, C_fp):
    DCF = Compute_Anormalized_DCF(matrix, pi, C_fn, C_fp)
    nDCF = Compute_Normalized_DCF(DCF, pi, C_fn, C_fp)
    return (DCF,nDCF)

def optimal_decisions_infpar(pi,C_fn,C_fp):
    LTE = numpy.load('Data/commedia_labels_infpar.npy')
    SPost = numpy.load('Data/commedia_llr_infpar.npy')
    threshold = -numpy.log( (pi*C_fn) / ( (1-pi) * C_fp ) )

    res = Discriminant_ratio(threshold,SPost)

    matrix =  confusion_matrix(LTE,res)
    print(matrix)

    DCF , nDCF = Compute_DCF(matrix, pi, C_fn, C_fp)
    print(DCF)
    print(nDCF)
    MinDCF = 1
    ROCPoints = []
    for val in numpy.sort(SPost):
        matrix = confusion_matrix(LTE, Discriminant_ratio(val, SPost))
        FNR = matrix[0][1] / (matrix[0][1] + matrix[1][1])
        ROCPoints.append( FNR )
        _ , tempDCF = Compute_DCF(matrix, pi, C_fn, C_fp)
        if(tempDCF < MinDCF):
            MinDCF = tempDCF
    print(f" Min : {MinDCF}")
    plt.figure()
    plt.plot(ROCPoints)
    plt.show()

if __name__ == '__main__':
    #LTE = numpy.load('Data/commedia_labels.npy')
    #SPost = numpy.load('Data/commedia_ll.npy').argmax(axis=0)

    #confusion_matrix(LTE,SPost)

    print("Confusion matrix with workpoint (0.5,1,1)")
    optimal_decisions_infpar(0.5,1,1)

    print("Confusion matrix with workpoint (0.8,1,1)")
    optimal_decisions_infpar(0.8,1,1)

    print("Confusion matrix with workpoint (0.5,10,1)")
    optimal_decisions_infpar(0.5,10,1)

    print("Confusion matrix with workpoint (0.8,1,10)")
    optimal_decisions_infpar(0.8,1,10)