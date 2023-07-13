import numpy
import scipy
import matplotlib.pyplot as pyplot
import sklearn.datasets

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1,v.size))

def dataCovarianceMatrix(D):
    mu = (D.mean(1))
    DC = D - vcol(mu)
    C = (1 / DC.shape[1]) * numpy.dot(DC, DC.T)
    return C , vcol(mu)

def within_class_covariance(D, N):
    C , _ = dataCovarianceMatrix(D)
    return C * D.size / N

def generalized_eigenvalue(SB,SW):
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:2]
    return W

def joint_diagonalization(Sb, Sw, m):
    Uw, sw, _ = numpy.linalg.svd(Sw)
    P1 = numpy.dot(Uw * vrow(1.0 / (sw ** 0.5)), Uw.T)
    Sbt = numpy.dot(P1, numpy.dot(Sb, P1.T))

    Ub, _, _ = numpy.linalg.svd(Sbt)
    P2 = Ub[:, 0:m]
    return numpy.dot(P1.T, P2)

def between_class_covariance(D, L):
    mu = vcol(D.mean(1))
    Sb = None
    for i in numpy.unique(L):
        mc = vcol(D[:, L == i].mean(1))
        mu_diff = mc - mu
        C = D[:, L == i].shape[1] * numpy.dot(mu_diff, mu_diff.T)
        if Sb is None:
            Sb = C
        else:
            Sb += C
    Sb /= D.shape[1]
    return Sb

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def load_digits():
    D, L = sklearn.datasets.load_digits()['data'].T, sklearn.datasets.load_digits()['target']
    return D, L