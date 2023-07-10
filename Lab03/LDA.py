import numpy
import matplotlib.pyplot as pyplot

import util

def vcol(v: numpy.array):
    return v.reshape((v.size, 1))

def LDA(DTR,LTR,m):
    Sb = util.between_class_covariance(DTR,LTR,3)
    Sw = numpy.zeros( (4,4) )
    for i in range(0,3):
        Sw += util.within_class_covariance(DTR[:,LTR == i],DTR.size)
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
    (DTR, LTR) = util.readfile_iris('./iris.csv')
    LDA(DTR,LTR,2)
