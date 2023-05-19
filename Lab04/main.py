import math

import matplotlib.pyplot as plt
import numpy

def vrow(v):
    return v.reshape((1,v.size))


def logpdf_GAU_ND_old(x, mu, C):
    _ , logC = numpy.linalg.slogdet(C)
    xc = x - mu
    a = - (x.shape[1]/2) * math.log(2 * math.pi)
    b = logC / 2
    c = numpy.dot(xc.T , numpy.dot(numpy.linalg.inv(C) , xc )) / 2
    N = a - b - c
    return N

def logpdf_GAU_ND(X, mu, C):
    _, log_determinant = numpy.linalg.slogdet(C)
    firstTerm = - (numpy.shape(X)[0] * 0.5) * math.log(2 * math.pi) - log_determinant * 0.5
    L = numpy.linalg.inv(C)
    XC = X - mu
    return firstTerm - 0.5 * (XC * numpy.dot(L,XC)).sum(0)


if __name__ == '__main__':
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    m = numpy.ones((1, 1)) * 1.0
    C = numpy.ones((1, 1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    plt.show()