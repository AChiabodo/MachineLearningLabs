import numpy
import util
from util import logpdf_GAU_ND
import scipy.special

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