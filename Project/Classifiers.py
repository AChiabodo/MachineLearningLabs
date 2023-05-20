import numpy
import util
from util import logpdf_GAU_ND
import scipy.special


# MultivariateGaussianClassifier
def linearMVG(DTR, LTR, DTE):
    hCls = {}
    n = numpy.unique(LTR).shape[0]
    for lab in numpy.unique(LTR):
        DCLS = DTR[:, LTR == lab]
        hCls[lab] = util.dataCovarianceMatrix(DCLS)
    ### Classification
    prior = util.vcol(numpy.ones(n) / n)
    S = []
    for hyp in numpy.unique(LTR):
        C, mu = hCls[hyp]
        fcond = numpy.exp(logpdf_GAU_ND(DTE, mu, C))
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S * prior
    P = SJoint / util.vrow(SJoint.sum(0))  # vrow(S.sum(0)) is the marginal

    return SJoint, P


def logMVG(DTR, LTR, DTE):
    hCls = {}
    n = numpy.unique(LTR).shape[0]
    for lab in numpy.unique(LTR):
        DCLS = DTR[:, LTR == lab]
        hCls[lab] = util.dataCovarianceMatrix(DCLS)
    # classification
    logprior = numpy.log(util.vcol(numpy.ones(n) / n))
    S = []
    for hyp in numpy.unique(LTR):
        C, mu = hCls[hyp]
        fcond = logpdf_GAU_ND(DTE, mu, C)
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S + logprior  # S is the logJoint
    logP = SJoint - util.vrow(scipy.special.logsumexp(SJoint, 0))
    # logsumexp does in a numerical stable way the sum and the exp for marginal
    P = numpy.exp(logP)
    return SJoint, P


def logNaiveMVG(DTR, LTR, DTE):
    hCls = {}
    n = numpy.unique(LTR).shape[0]
    for lab in numpy.unique(LTR):
        DCLS = DTR[:, LTR == lab]
        C, mu = util.dataCovarianceMatrix(DCLS)
        ones = numpy.diag(numpy.ones(DCLS.shape[0]))
        hCls[lab] = (C * ones, mu)

    logprior = numpy.log(util.vcol(numpy.ones(n) / n))
    S = []
    for hyp in numpy.unique(LTR):
        C, mu = hCls[hyp]
        fcond = logpdf_GAU_ND(DTE, mu, C)
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S + logprior  # S is the logJoint
    logP = SJoint - util.vrow(scipy.special.logsumexp(SJoint, 0))
    # logsumexp does in a numerical stable way the sum and the exp for marginal
    P = numpy.exp(logP)

    return SJoint, P


def logTiedMVG(DTR, LTR, DTE):
    hCls = {}
    n = numpy.unique(LTR).shape[0]
    Sw = numpy.zeros((DTR.shape[0], DTR.shape[0]))
    for i in range(0, n):
        Sw += util.within_class_covariance(DTR[:, LTR == i], DTR.size)

    for lab in numpy.unique(LTR):
        DCLS = DTR[:, LTR == lab]
        _, mu = util.dataCovarianceMatrix(DCLS)
        hCls[lab] = (Sw, mu)

    logprior = numpy.log(util.vcol(numpy.ones(n) / n))
    S = []
    for hyp in numpy.unique(LTR):
        C, mu = hCls[hyp]
        fcond = logpdf_GAU_ND(DTE, mu, C)
        S.append(util.vrow(fcond))
    S = numpy.vstack(S)
    SJoint = S + logprior  # S is the logJoint
    logP = SJoint - util.vrow(scipy.special.logsumexp(SJoint, 0))
    # logsumexp does in a numerical stable way the sum and the exp for marginal
    P = numpy.exp(logP)

    return SJoint, P


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
        loss_per_sample = numpy.logaddexp(0, -self.ZTR * scores)
        loss = 0.5 * self.l * numpy.linalg.norm(w) ** 2 + loss_per_sample.mean()
        return loss

    def train(self):
        x0 = numpy.zeros(self.DTR.shape[0] + 1)
        xOpt, fOpt, d = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0=x0, approx_grad=True)
        w, b = util.vcol(xOpt[0:self.DTR.shape[0]]), xOpt[-1]
        return w, b

    def evaluate(self, DTE, LTE):
        w, b = self.train()
        Score = numpy.dot(w.T, DTE) + b
        PLabel = (Score > 0).astype(int)
        Error = ((LTE != PLabel).astype(int).sum() / DTE.shape[1]) * 100
        return Error, w, b

    def confusion_matrix(self, DTE, LTE):
        w, b = self.train()
        Score = numpy.dot(w.T, DTE) + b
        PLabel = (Score > 0).astype(int)
        return util.confusion_matrix(LTE, PLabel)
