import csv
import operator
import time
import numpy as np

def loadData():
    trainList = []
    with open('diabetes.csv') as datafile:
        lines = csv.reader(datafile)
        for line in lines:
            trainList.append(line)
    trainList.remove(trainList[0])
    trainList = np.mat(np.array(trainList))
    m,n = np.shape(trainList)
    data = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
                data[i,j] = float(trainList[i,j])
    return data

def featureScaling(dataSet):
    maxData = dataSet.max(0)
    dataSet = dataSet/maxData
    return dataSet

# This knn classifier includes three different distance matrix
def knn(testData, trainDataSet, trainLabel):
    trainSize = trainDataSet.shape[0]
    tileTestData = np.tile(testData, (trainSize,1))
    # 1 : Euclidean Distance Model
    diffMat1 = tileTestData - trainDataSet
    sqDiffMat = np.array(diffMat1)**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances1 = sqDistances**0.5
    sortedDistIndex1 = distances1.argsort()
    countLabel1 = {}
    
    # 2 : Cosine Distance Model
    diffMat2 = np.multiply(tileTestData, trainDataSet)
    prodSum = diffMat2.sum(axis=1)
    testSq = np.array(testData)**2
    testSqSum = testSq.sum()
    testSqSumRt = testSqSum**0.5
    testSqSumRt = np.tile(testSqSumRt, (1,trainSize))
    trainSq = np.array(trainDataSet)**2
    trainSqSum = trainSq.sum(axis=1)
    trainSqSumRt = trainSqSum**0.5
    distances2 = np.divide(prodSum, (np.multiply(testSqSumRt,trainSqSumRt)))
    sortedDistIndex2 = distances2[0].argsort()
    countLabel2 = {}

    # 3 : Manhattan Distance Model
    distances3 = tileTestData - trainDataSet
    distances3 = np.fabs(distances3)
    distances3 = distances3.sum(axis=1)
    sortedDistIndex3 = distances3.argsort()
    countLabel3 = {}

    for i in range(k):
        curLabel1 = trainLabel[sortedDistIndex1[i]]
        countLabel1[curLabel1] = countLabel1.get(curLabel1, 0) + 1
        curLabel2 = trainLabel[sortedDistIndex2[i]]
        countLabel2[curLabel2] = countLabel2.get(curLabel2, 0) + 1
        curLabel3 = trainLabel[sortedDistIndex3[i]]
        countLabel3[curLabel3] = countLabel3.get(curLabel3, 0) + 1
    
    rankedLabel1 = sorted(countLabel1.items(), key=operator.itemgetter(1), reverse=True)
    rankedLabel2 = sorted(countLabel2.items(), key=operator.itemgetter(1), reverse=True)
    rankedLabel3 = sorted(countLabel3.items(), key=operator.itemgetter(1), reverse=True)
    res = []
    res.append(int(rankedLabel1[0][0]))
    res.append(int(rankedLabel2[0][0]))
    res.append(int(rankedLabel3[0][0]))
    return res


# ------loading data----------
start1 = time.perf_counter()
trainData = loadData()
size,num = trainData.shape
testData = featureScaling(trainData[int(size*0.8):,:(num-2)])
testLabel = trainData[int(size*0.8):,num-1]
trainLabel = trainData[:int(size*0.8),num-1]
trainData = featureScaling(trainData[:int(size*0.8),:(num-2)])
# ----------------------------


k = 5
testSize = testData.shape[0]
# 1 : Euclidean Distance Model
# 2 : Cosine Distance Model
# 3 : Manhattan Distance Model
errorCount1 = 0
errorCount2 = 0
errorCount3 = 0
confusionMat1 = np.zeros((2,2))
confusionMat2 = np.zeros((2,2))
confusionMat3 = np.zeros((2,2))
start2 = time.perf_counter()
for i in range(testSize):
    curRes = knn(testData[i], trainData, trainLabel)
    trueRes = int(testLabel[i])
    if(curRes[0] != trueRes):
        errorCount1 += 1
    if(curRes[1] != trueRes):
        errorCount2 += 1
    if(curRes[2] != trueRes):
        errorCount3 += 1
    confusionMat1[curRes[0],trueRes] += 1
    confusionMat2[curRes[1],trueRes] += 1
    confusionMat3[curRes[2],trueRes] += 1
end2 = time.perf_counter()
print("")
print("------Euclidean Distance Model------------") 
print("Confusion Matrix:")
print(confusionMat1)
print("accurancy is: %.2f%%" %(100*(testSize-errorCount1)/float(testSize)))
print("")
print("------Cosine Distance Model------------") 
print("Confusion Matrix:")
print(confusionMat2)
print("accurancy is: %.2f%%" %(100*(testSize-errorCount2)/float(testSize)))
print("")
print("------Manhattan Distance Model------------") 
print("Confusion Matrix:")
print(confusionMat3)
print("The accurancy is: %.2f%%" %(100*(testSize-errorCount3)/float(testSize)))
print("Total runtime is ",time.perf_counter()-start1)
print("Actual wall time is ",end2-start2)
