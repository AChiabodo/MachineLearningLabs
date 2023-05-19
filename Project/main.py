import numpy
import util
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
                attrs = line.split(',')[0:4]
                attrs = util.vcol(numpy.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass
        D , L = numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)
        return D, L

def logpdf_GAU_ND(X, mu, C):
    _, log_determinant = numpy.linalg.slogdet(C)
    firstTerm = - (numpy.shape(X)[0] * 0.5) * numpy.log(2 * numpy.pi) - log_determinant * 0.5
    L = numpy.linalg.inv(C)
    XC = X - mu
    return firstTerm - 0.5 * (XC * numpy.dot(L,XC)).sum(0)

if __name__ == '__main__':

    # Load the dataset
    D, L = readfile()

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
    print('Accuracy:', accuracy)