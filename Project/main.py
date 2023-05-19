import numpy
import util
from util import logpdf_GAU_ND
from Classifiers import logMVG , linearMVG , logTiedMVG , logNaiveMVG
from scipy.stats import multivariate_normal

def readfile():
    DList = []
    labelsList = []
    hLabels = {
        '0': 0,
        '1': 1
    }

    with open('data/Train.csv') as f:
        for line in f:
            try:
                attrs = line.split(',')[0:10]
                attrs = util.vcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
        D , L = numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
        return D, L

def __olds():
    # Compute the mean and covariance for each class
    means = []
    covs = []
    for label in numpy.unique(L):
        temp = D[:,L == label]
        cov, mean = util.dataCovarianceMatrix(D)
        means.append(mean)
        covs.append(cov)

    # Compute the log-likelihood for each class
    log_likelihoods = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        log_likelihood = logpdf_GAU_ND(D, mean, cov)
        log_likelihoods.append(log_likelihood)

    # Predict the class with the highest log-likelihood
    y_pred = numpy.argmax(log_likelihoods, axis=0)
    # Compute the accuracy
    accuracy = numpy.mean(y_pred == L)

if __name__ == '__main__':

    # Load the dataset
    D, L = readfile()
    (DTR, LTR), (DTE, LTE) = util.split_db_2to1(D, L)

    _, P = logMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100
    print(error)

    _, P = logNaiveMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100
    print(error)

    _, P = logTiedMVG(DTR, LTR, DTE)
    SPost = P.argmax(axis=0)
    error = (SPost - LTE).sum() / LTE.shape[0] * 100
    print(error)