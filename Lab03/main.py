import util





if __name__ == '__main__':
    (DTR, LTR) = util.readfile_iris('./iris.csv')
    W = util.PCA(DTR,LTR,2)
