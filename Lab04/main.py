import math

import matplotlib.pyplot as plt
import numpy
import util

def vrow(v):
    return v.reshape((1, v.size))


def logpdf_GAU_ND(X, mu, C):
    _, log_determinant = numpy.linalg.slogdet(C)
    firstTerm = - (numpy.shape(X)[0] * 0.5) * math.log(2 * math.pi) - log_determinant * 0.5
    L = numpy.linalg.inv(C)
    XC = X - mu
    return firstTerm - 0.5 * (XC * numpy.dot(L, XC)).sum(0)

def loglikelihood(XND, m_ML, C_ML):
    return logpdf_GAU_ND(XND, m_ML, C_ML).sum()

if __name__ == '__main__':
    plt.figure()
    XPlot = numpy.linspace(-8, 12, 1000)
    m = numpy.ones((1, 1)) * 1.0
    C = numpy.ones((1, 1)) * 2.0
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    #plt.show()

    pdfSol = numpy.load('data/llGAU.npy')
    pdfGau = logpdf_GAU_ND(vrow(XPlot), m, C)
    assert (numpy.abs(pdfSol - pdfGau).max() < 1e-10)

    XND = numpy.load('data/XND.npy')
    mu = numpy.load('data/muND.npy')
    C = numpy.load('data/CND.npy')
    pdfSol = numpy.load('data/llND.npy')
    pdfGau = logpdf_GAU_ND(XND, mu, C)
    assert (numpy.abs(pdfSol - pdfGau).max() < 1e-10)

    # Maximum Likelihood Estimation
    ## The ML estimate for the parameters of a Multivariate Gaussian distribution correspond to the empirical dataset mean and the empirical dataset covariance
    XND = numpy.load('data/XND.npy')
    C_ML , mu_ML = util.dataCovarianceMatrix(XND)

    ## We can also compute the log-likelihood for our estimates. The log-likelihood corresponds to the sum of the log-density computed over all the samples
    ll = loglikelihood(XND, mu_ML, C_ML)
    print(ll)

    X1D = numpy.load('data/X1D.npy')
    C_ML , mu_ML = util.dataCovarianceMatrix(X1D)

    ## We can visualize how well the estimated density fits the samples plotting both the histogram of the samples and the density
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = numpy.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), mu_ML - 0.1, C_ML)))
    plt.show()

    ## We can also verify that computing the log-likelihood for other values of μ and Σ would results in a lower value of the log-likelihood.
    ll = loglikelihood(X1D, mu_ML, C_ML)
    ll_moved = loglikelihood(X1D, mu_ML - 0.1, C_ML + 0.4)
    print(abs(ll_moved) - abs(ll)) ##this should be positive -> the log-likelihood is higher for the correct ML estimate
