# This is a sample Python script.
import numpy as numpy
import matplotlib.pyplot as plt
import util
import LDA
from matplotlib.backends.backend_pdf import PdfPages

def PCA(DTR,m):
    C = util.dataCovarianceMatrix(DTR)
    s, U = numpy.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    W = numpy.dot(P.T, DTR)
    return W

def readfile():
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }

    with open('iris.csv') as f:
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
        C = util.dataCovarianceMatrix(D)


        s, U = numpy.linalg.eigh(C)
        PCA = numpy.load('IRIS_PCA_matrix_m4.npy')
        print(abs(PCA) - abs(U[:,::-1]))
        for m in range(1,4):
            P = U[:, ::-1][:, 0:m]
            DP = numpy.dot(P.T, D)
            print(DP.shape)
            plot_scatter(DP,L , m)

def plot_scatter(D, L , m):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    D2 = D[:, L == 2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    for dIdx1 in range(m):
        for dIdx2 in range(m):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label='Setosa')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label='Versicolor')
            plt.scatter(D2[dIdx1, :], D2[dIdx2, :], label='Virginica')
            plt.title(f"plot : {m}")
            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            #plt.savefig('scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()

def plot_hists(DTR: numpy.array, LTR: numpy.array):
    f0 = DTR[:, LTR == 0]
    f1 = DTR[:, LTR == 1]
    f2 = DTR[:, LTR == 2]

    hFea = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
    }

    with PdfPages('target/hists.pdf') as pdf:
        for i in range(DTR.shape[0]):
            fig = plt.figure()
            plt.xlabel(f"Feature: {i}")
            plt.ylabel("Density")
            plt.hist(f0[i, :], density=True, bins=20, label='Iris-setosa', alpha=0.4)
            plt.hist(f1[i, :], density=True, bins=20, label='Iris-versicolor', alpha=0.4)
            plt.hist(f2[i, :], density=True, bins=20, label='Iris-virginica', alpha=0.4)

            plt.legend()
            pdf.savefig(fig)
            plt.close(fig)


if __name__ == '__main__':
    DTR , LTR = util.readfile_iris('./iris.csv')
    W = PCA(DTR,1)
    plot_hists(DTR,LTR)
    plot_hists(W,LTR)
