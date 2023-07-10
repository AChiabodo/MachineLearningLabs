import numpy
import scipy

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
    return C

def within_class_covariance(D, N):
    return dataCovarianceMatrix(D) * D.size / N

def generalized_eigenvalue(SB,SW):
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:2]
    return W

def between_class_covariance(D, L, K):
    mu = vcol(D.mean(1))
    Sb = numpy.zeros((4, 4))
    for i in range(0, K):
        mc = vcol(D[:, L == i].mean(1))
        mu_diff = mc - mu
        C = D[:, L == i].shape[1] * numpy.dot(mu_diff, mu_diff.T)
        Sb += C
    Sb /= D.shape[1]
    return Sb
