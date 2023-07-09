import numpy
import scipy as scipy
import matplotlib.pyplot as pyplot

import util


def within_class_covariance(D, N):
    return util.dataCovarianceMatrix(D) * D.size / N

def generalized_eigenvalue(SB,SW):
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:2]
    return W

def between_class_covariance(D, L, K):
    mu = util.vcol(D.mean(1))
    Sb = numpy.zeros((4, 4))
    for i in range(0, K):
        mc = util.vcol(D[:, L == i].mean(1))
        mu_diff = mc - mu
        C = D[:, L == i].shape[1] * numpy.dot(mu_diff, mu_diff.T)
        Sb += C
    Sb /= D.shape[1]
    return Sb


def lda_old():
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

    with open('iris.csv') as f:
        for line in f:
            try:
                attrs = line.split(',')[0:4]
                attrs = util.vcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
        D, L = numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
        Sb = between_class_covariance(D, L, 3)
        Sw = numpy.zeros((4, 4))
        for i in range(0, 3):
            Sw += within_class_covariance(D[:, L == i], D.size)
        #print(Sw)
        #print(Sb)
        W = generalized_eigenvalue(Sb,Sw)
        m = 2
        U, s, _ = numpy.linalg.svd(Sw)
        P1 = numpy.dot(U * util.vrow(1.0 / (s ** 0.5)), U.T)
        SBTilde =numpy.dot(P1,numpy.dot(Sb,P1.T))
        U,_,_ = numpy.linalg.svd(SBTilde)
        P2 = U[:,0:m]
        W = numpy.dot(P1.T,P2)

def readfile(file):
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

def vcol(v: numpy.array):
    return v.reshape((v.size, 1))

def LDA(DTR,LTR,m):
    Sb = between_class_covariance(DTR,LTR,3)
    Sw = numpy.zeros( (4,4) )
    for i in range(0,3):
        Sw += within_class_covariance(DTR[:,LTR == i],DTR.size)
    test = numpy.load('IRIS_LDA_matrix_m2.npy')
    U, s, _ = numpy.linalg.svd(Sw)
    P1 = numpy.dot(U * util.vrow(1.0 / (s ** 0.5)), U.T)
    SBTilde = numpy.dot(P1, numpy.dot(Sb, P1.T))
    U, _, _ = numpy.linalg.svd(SBTilde)
    P2 = U[:, 0:m]
    W = numpy.dot(P1.T, P2)
    print(W / test)
def analize(DTR,LDR):
    setosa = DTR[(LDR == 0).reshape(150), 2]
    versicolor = DTR[(LDR == 1).reshape(150), 2]
    virginica = DTR[(LDR == 2).reshape(150), 2]
    # pyplot.plot(numpy.arange(0,setosa.size),setosa)
    pyplot.hist(setosa, bins=15, color='blue', label="setosa")
    pyplot.hist(versicolor, bins=15, color='red')
    pyplot.hist(virginica, bins=15, color='yellow')
    pyplot.show()

if __name__ == '__main__':
    (DTR, LTR) = readfile('./iris.csv')
    LDA(DTR,LTR,2)
