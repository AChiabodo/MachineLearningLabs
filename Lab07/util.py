import numpy

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1,v.size))

def dataCovarianceMatrix(D):
    mu = (D.mean(1))
    DC = D  - vcol(mu)
    C = numpy.dot(DC, DC.T)/DC.shape[1]
    return C , vcol(mu)

def within_class_covariance(D, N):
    return dataCovarianceMatrix(D)[0] * D.size / N