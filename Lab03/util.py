import numpy

def vcol(v):
    return v.reshape((v.size, 1))

def vrow(v):
    return v.reshape((1,v.size))

def dataCovarianceMatrix(D):
    mu = (D.mean(1))
    DC = D - vcol(mu)
    C = (1 / DC.shape[1]) * numpy.dot(DC, DC.T)
    return C
