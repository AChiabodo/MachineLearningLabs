import load
import numpy
if __name__ == '__main__':
    # Load the tercets and split the lists in training and test lists

    Inf_count = []
    Pur_count = []
    Par_count = []
    dict = {}
    current = 0
    lInf, lPur, lPar = load.load_data()

    lInf_train, lInf_evaluation = load.split_data(lInf, 4)
    lPur_train, lPur_evaluation = load.split_data(lPur, 4)
    lPar_train, lPar_evaluation = load.split_data(lPar, 4)

    for string in lInf_train:
        for word in string.split(" "):
            if word in dict :
                Inf_count[dict[word]] = Inf_count[dict[word]] +1
            else :
                dict[word] = current
                Inf_count.append(1)
                Pur_count.append(0)
                Par_count.append(0)
                current = current + 1
    for string in lPur_train:
        for word in string.split(" "):
            if word in dict :
                Pur_count[dict[word]] = Inf_count[dict[word]] +1
            else :
                dict[word] = current
                Inf_count.append(0)
                Pur_count.append(1)
                Par_count.append(0)
                current = current + 1

    for string in lPar_train:
        for word in string.split(" "):
            if word in dict :
                Par_count[dict[word]] = Inf_count[dict[word]] +1
            else :
                dict[word] = current
                Inf_count.append(0)
                Pur_count.append(0)
                Par_count.append(1)
                current = current + 1
    P = []
    P2 = []
    epsilon = 0.000001
    tot = sum(Inf_count)
    P.append( list(map(lambda n: numpy.log(n / tot + epsilon), Inf_count)) )
    P2.append( sum(list(map(lambda n: numpy.log(n / tot + epsilon), Inf_count))) )
    tot = sum(Pur_count)
    P.append(list(map(lambda n: numpy.log(n / tot  + epsilon), Pur_count)))
    P2.append(sum(list(map(lambda n: numpy.log(n / tot + epsilon), Pur_count))))
    tot = sum(Par_count)
    P.append( list(map(lambda n: numpy.log(n / tot + epsilon), Par_count)))
    P2.append(sum(list(map(lambda n: numpy.log(n / tot + epsilon), Par_count))))

    true = 0

    for string in lInf_evaluation:
        test = [0, 0, 0]
        for word in string.split(" "):
            if word in dict:
                for i in range(3):
                    test [i] = test[i] + P[i][dict[word]]
        if(test.index(max(test)) == 0):
            true = true + 1
    print(true / lInf_evaluation.__len__())



