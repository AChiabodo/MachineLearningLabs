#try to find the minimizer of a function
import numpy
import numpy as np
import scipy
import sklearn.datasets

def vcol(v) -> numpy.array:
    return v.reshape((v.size, 1))

def vrow(v) -> numpy.array:
    return v.reshape((1, v.size))

def f ( x: np.array) :
    y = x[0]
    z = x[1]
    return (y+3)**2 + np.sin(y) + (z+1)**2

def explicit_gradient(x: np.array ) :
    y = x[0]
    z = x[1]
    return numpy.array([ (2*y + 6 + np.cos(y)) , (2*z + 2)])

def try_minimizer() :
    x = np.array([0, 17])
    estmin, objmin, info = scipy.optimize.fmin_l_bfgs_b(f, x, approx_grad=True, iprint=1)
    print(estmin)
    print(objmin)
    print(info)
    print("---------------------------------------------------------------------------------------")
    estmin, objmin, info = scipy.optimize.fmin_l_bfgs_b(f, x, explicit_gradient, iprint=1)
    print(estmin)
    print(objmin)
    print(info)


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L


def logreg_obj( v : np.array, lambda_v : float , DTR :np.array , LTR : np.array) :
    w = vcol(v[ 0 : -1])
    b = v[ -1 ]
    regolarization_term = lambda_v / 2 * (np.dot(w.T,w))
    logistic_loss = 0
    for i in range(DTR.shape[1]) :
        zi_si = (2*LTR[i]-1) * ((numpy.dot(w.T, vcol(DTR[:, i])) + b ))
        logistic_loss += np.logaddexp( 0,- zi_si)

    return regolarization_term + logistic_loss / DTR.shape[1]

def min_logregr_loss(DTE,LTE,v) :
    accuracy = 0
    LP = []
    for i in range(DTE.shape[1]):
        score = np.dot(vrow(v[0:4]), vcol(DTE[:, i])) + v[4]
        if score > 0:
            LP.append(1)
        else:
            LP.append(0)

        if LP[i] == LTE[i]:
            accuracy += 1

    print("Accuracy")
    print(accuracy / DTE.shape[1])
    print(" Error ")
    print((DTE.shape[1] - accuracy) / DTE.shape[1])

if __name__ == '__main__':
    #try_minimizer()
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    lambda_v = 1
    x0 = np.zeros(DTR.shape[0] + 1)
    estmin, objmin, info = scipy.optimize.fmin_l_bfgs_b(logreg_obj, x0, args=(lambda_v, DTR, LTR), approx_grad=True,
                                                        iprint=1)
    print(estmin)
    print(objmin)
    print(info)
    min_logregr_loss(DTE, LTE, estmin)