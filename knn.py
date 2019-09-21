import numpy
import time
import csv
import operator
from numpy import*
def loadTrainData(): 
    l=[] 
    with open("train.csv") as myFile: 
        lines=csv.reader(myFile) 
        for line in lines: 
            l.append(line) 
        l.remove(l[0])
        l=array(l)
        label=l[:,0] 
        data=l[:,1:] 
    return nomalizing(toInt(data)),toInt(label)

def toInt(array): 
    array=mat(array) 
    m,n=shape(array) 
    newArray=zeros((m,n)) 
    for i in range(m): 
        for j in range(n): 
            newArray[i,j]=int(array[i,j]) 
            return newArray 

def nomalizing(array):
    m,n=shape(array) 
    for i in range(m): 
        for j in range(n):
            if array[i,j]!=0:
                array[i,j]=1 
            return array


def classify(inX, dataSet, labels, k): 
    inX=mat(inX) 
    dataSet=mat(dataSet)        
    labels=mat(labels) 
    dataSetSize = dataSet.shape[0]      #number of data
    diffMat = tile(inX, (dataSetSize,1)) - dataSet      #m*n matrix
    sqDiffMat = array(diffMat)**2 
    sqDistances = sqDiffMat.sum(axis=1)         ##plus line by line
    distances = sqDistances**0.5 
    sortedDistIndicies = distances.argsort() 
    classCount={} 
    for i in range(k): 
        voteIlabel = labels[0,sortedDistIndicies[i]] 
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) 
    return sortedClassCount[0][0]


def handwritingClassTest(): 
    timeStart = time.perf_counter()
    trainData,trainLabel=loadTrainData()
    size = trainData.shape[0] 
    testLabel = trainLabel[:,int(size*0.8):]
    trainLabel = trainLabel[:,:int(size*0.8)]
    testData = trainData[int(size*0.8):,:]
    
    print(testLabel)
    trainData = trainData[:int(size*0.8),:]
    testSize = testData.shape[0]
    k = 3
    errorNumber = 0
    confusionMatrix = zeros((10,10))
    startTime = time.perf_counter()
    for i in range(testSize):
        print("This predicted #", i + 1, " test data ")
        currentResult = classify(testData[i], trainData, trainLabel,3)
        actualResult = int(testLabel[0,i])
        if(currentResult != actualResult):
            errorNumber = errorNumber + 1
        confusionMatrix[int(currentResult),int(actualResult)] += 1
    print("Confusion Matrix")
    print(confusionMatrix)
    print("The accurancy rate is: %f" %((testSize - errorNumber)/float(testSize)))
    print("Running time is", time.perf_counter() - timeStart)
    print("Actual running time is", time.perf_counter() - startTime)
        
handwritingClassTest()
