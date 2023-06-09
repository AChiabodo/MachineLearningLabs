import math
import time

import numpy
import scipy.special
import sklearn.datasets
import util
import matplotlib.pyplot as plt

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

def load_digits():
    D, L = sklearn.datasets.load_digits()['data'].T, sklearn.datasets.load_digits()['target']
    return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]    #Data Training
    DTE = D[:, idxTest]     #Data Test
    LTR = L[idxTrain]       #Label Training
    LTE = L[idxTest]        #Label Test
    return (DTR, LTR), (DTE, LTE)

def split_in_k(D,L,k, seed=0):
    #we want to mantain the K (#of models / groups) as large as possible to emulate the "leave-one-out"
    #for IRIS is ok to implement the leave-one-out
    #D, L = load_iris()

    k_folds = []
    label_k_folds = []
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxSplit = numpy.array_split(idx, k)
    for i in range(k):
        k_folds.append(D[:, idxSplit[i]])
        label_k_folds.append(L[idxSplit[i]])
    return k_folds , label_k_folds

def confusion_matrix(LTE,SPost):
    n = numpy.unique(LTE).shape[0]
    matrix = numpy.zeros([n,n])
    for i in numpy.unique(LTE):
        for j in numpy.unique(LTE):
            matrix[i][j] = (SPost[LTE == j] == i).sum()
    return matrix

def Discriminant_ratio(threshold,SPost):
    res = numpy.copy(SPost)
    res[SPost > threshold] = 1
    res[SPost <= threshold] = 0
    return (SPost > threshold).astype(int)

def optimal_decisions(workPoint):
    """

    :param WorkPoint:
    :param LTE:
    :param SPost:
    """
    pi, C_fn, C_fp = workPoint
    LTE = numpy.load('Data/commedia_labels_infpar.npy')
    SPost = numpy.load('Data/commedia_llr_infpar.npy')
    print(SPost.shape)
    threshold = -numpy.log( (pi*C_fn) / ( (1-pi) * C_fp ) )

    res = Discriminant_ratio(threshold,SPost)

    matrix = confusion_matrix(LTE,res)
    #print(matrix) print(DCF) print(nDCF)

    DCF , nDCF = util.Compute_DCF(matrix, pi, C_fn, C_fp)

    MinDCF = 1
    FPRs = []
    TPRs = []
    for val in numpy.sort(SPost):
        matrix = confusion_matrix(LTE, Discriminant_ratio(val, SPost))
        FPR = matrix[1][0] / (matrix[0][0] + matrix[1][0])
        FNR = matrix[0][1] / (matrix[0][1] + matrix[1][1])
        FPRs.append( FPR )
        TPRs.append( 1 - FNR)
        _ , tempDCF = util.Compute_DCF(matrix, pi, C_fn, C_fp)
        if(tempDCF < MinDCF):
            MinDCF = tempDCF
    print(f" Min : {MinDCF}")
    plot_roc_curve(FPRs, TPRs)
    minDCF = compute_minDCF(LTE,SPost,WorkPoint(pi,C_fn,C_fp))
    print(f" Min : {minDCF}")

def plot_roc_curve(FPRs, TPRs):
    plt.figure()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.plot(FPRs, TPRs)
    plt.grid()
    plt.show()


class WorkPoint:
    def __init__(self, pi: float, C_fn: float, C_fp: float):
        self.pi = pi
        self.C_fn = C_fn
        self.C_fp = C_fp

    def effective_prior(self):
        return (self.pi * self.C_fn) / (self.pi * self.C_fn + (1 - self.pi) * self.C_fp)


def compute_minDCF(LTE,SPost,workPoint):
    idx = numpy.argsort(SPost)
    sortL = LTE[idx]
    MinDCF = 1
    startingMatrix = confusion_matrix(LTE, Discriminant_ratio(-math.inf, SPost))
    for val in sortL:
        if(val == 0):
            startingMatrix[0][0] = startingMatrix[0][0] + 1
            startingMatrix[1][0] = startingMatrix[1][0] - 1
        else:
            startingMatrix[0][1] = startingMatrix[0][1] + 1
            startingMatrix[1][1] = startingMatrix[1][1] - 1
        _, tempDCF = util.Compute_DCF(startingMatrix, workPoint.pi, workPoint.C_fn, workPoint.C_fp)
        if (tempDCF < MinDCF):
            MinDCF = tempDCF
    return MinDCF

if __name__ == '__main__':
    #LTE = numpy.load('Data/commedia_labels.npy')
    #SPost = numpy.load('Data/commedia_ll.npy').argmax(axis=0)
    print("Confusion matrix with workpoint (0.5,1,1)")
    optimal_decisions([0.5, 1, 1])

    #confusion_matrix(LTE,SPost)

def temp():
    print("Confusion matrix with workpoint (0.5,1,1)")
    optimal_decisions([0.5, 1, 1])

    print("Confusion matrix with workpoint (0.8,1,1)")
    optimal_decisions([0.8, 1, 1])

    print("Confusion matrix with workpoint (0.5,10,1)")
    optimal_decisions([0.5, 10, 1])

    print("Confusion matrix with workpoint (0.8,1,10)")
    optimal_decisions([0.8, 1, 10])