import util
import numpy

def PCA(DTR,m):
    C , _ = util.dataCovarianceMatrix(DTR)
    _ , U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    #U, _, _ = numpy.linalg.svd(C)
    #P = U[:, 0:m]
    W = numpy.dot(P.T, DTR)
    return W

def LDA(DTR,LTR,m=None):
    if m is None:
        m = numpy.unique(LTR).size - 1
    Sb = util.between_class_covariance(DTR,LTR)
    Sw = None
    for i in range(0,numpy.unique(LTR).size):
        if Sw is None:
            Sw = util.within_class_covariance(DTR[:,LTR == i],DTR.size)
        else:
            Sw += util.within_class_covariance(DTR[:,LTR == i],DTR.size)
    U, s, _ = numpy.linalg.svd(Sw)
    P1 = numpy.dot( numpy.dot(U, numpy.diag(1.0/(s**0.5))), U.T )
    SBTilde = numpy.dot(P1, numpy.dot(Sb, P1.T))
    U, _, _ = numpy.linalg.svd(SBTilde)
    P2 = U[:, 0:m]
    W = numpy.dot(P1.T, P2)
    return numpy.dot(W.T,DTR)

if __name__ == '__main__':
    print("Testing DimensionalityReduction.py")
    DTR , LTR = util.load_iris()
    W = PCA(DTR,4)
    Test_PCA = numpy.dot(numpy.load('data/IRIS_PCA_matrix_m4.npy').T,DTR)
    if numpy.allclose(W,Test_PCA):
        print("PCA Test Passed")
    W = abs(LDA(DTR,LTR,2))
    Test_LDA = abs(numpy.dot(numpy.load('data/IRIS_LDA_matrix_m2.npy').T,DTR))
    if numpy.allclose(W,Test_LDA):
        print("LDA Test Passed")
    print("Testing Complete")