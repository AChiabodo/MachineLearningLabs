import numpy
import matplotlib.pyplot as pyplot
# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def analize():
    vector = numpy.zeros((150,4))
    label = numpy.zeros((150,1))
    with open('iris.csv') as file:
        for (i,line) in enumerate(file):
            temp = line.rstrip().split(",")
            vector[i] = temp[0:4]
            if temp[4] == "Iris-setosa":
                temp = 0
            elif temp[4] == "Iris-versicolor":
                temp = 1
            else:
                temp = 2
            label[i] = temp
        #print(numpy.hstack([vector , label]))

        setosa = vector[(label ==0).reshape(150), 2]
        versicolor = vector[(label ==1).reshape(150), 2]
        virginica = vector[(label ==2).reshape(150), 2]
        #pyplot.plot(numpy.arange(0,setosa.size),setosa)
        pyplot.hist(setosa,bins=15,color='blue',label="setosa")
        pyplot.hist(versicolor, bins=15,color='red')
        pyplot.hist(virginica, bins=15,color='yellow')
        pyplot.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    analize()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
