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

def Compute_Anormalized_DCF(matrix, pi, C_fn, C_fp):
    FNR = matrix[0][1] / (matrix[0][1] + matrix[1][1])
    FPR = matrix[1][0] / (matrix[0][0] + matrix[1][0])

    DCF = pi * C_fn * FNR + (1-pi) * C_fp * FPR
    return DCF

def Compute_Normalized_DCF(DCF,pi,C_fn,C_fp):
    optimal_risk = numpy.min([pi * C_fn, (1 - pi) * C_fp])
    return DCF / optimal_risk

def Compute_DCF(matrix, pi, C_fn, C_fp):
    DCF = Compute_Anormalized_DCF(matrix, pi, C_fn, C_fp)
    nDCF = Compute_Normalized_DCF(DCF, pi, C_fn, C_fp)
    return (DCF,nDCF)

def logpdf_GAU_ND(X, mu, C):
    _, log_determinant = numpy.linalg.slogdet(C)
    firstTerm = - (numpy.shape(X)[0] * 0.5) * numpy.log(2 * numpy.pi) - log_determinant * 0.5
    L = numpy.linalg.inv(C)
    XC = X - mu
    return firstTerm - 0.5 * (XC * numpy.dot(L,XC)).sum(0)

def effective_prior(Cmiss,Cfa,pTar):
    pTar * Cmiss/(pTar*Cmiss + (1-pTar)*Cfa)