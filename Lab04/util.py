import numpy
import scipy
import matplotlib.pyplot as pyplot
import sklearn.datasets


def readfile_iris(file):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica' : 2
    }

    with open(file) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:4]
                attrs = vcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
        D, L = numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
        return D, L

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

def analize(DTR,LDR):
    setosa = DTR[(LDR == 0).reshape(150), 2]
    versicolor = DTR[(LDR == 1).reshape(150), 2]
    virginica = DTR[(LDR == 2).reshape(150), 2]
    # pyplot.plot(numpy.arange(0,setosa.size),setosa)
    pyplot.hist(setosa, bins=15, color='blue', label="setosa")
    pyplot.hist(versicolor, bins=15, color='red')
    pyplot.hist(virginica, bins=15, color='yellow')
    pyplot.show()
