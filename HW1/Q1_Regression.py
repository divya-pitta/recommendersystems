import numpy as np
import urllib.request
from collections import Counter
import matplotlib.pyplot as plt
import json
import scipy.optimize
import random

def getData(dataUrl):
    for line in urllib.request.urlopen(dataUrl):
        #print(line)
        yield eval(line)

#Reading the data from url line by line and creating a list out of it
print("Reading data")
data = list(getData("http://jmcauley.ucsd.edu/cse258/data/beer/beer_50000.json"))
print(data[0])

def feature(datum):
    feat = [1]
    return feat

BeerStyle = [d['beer/style'] for d in data]
#for b in BeerStyle:
#    print(b)
for b in Counter(BeerStyle):
    print(b)
    BeerStyleTaste = [d['review/taste'] for d in data if d['beer/style'] == b]
    #print(np.mean(np.array(BeerStyleTaste).astype(np.float)))
    print(np.average(BeerStyleTaste))
print(Counter(BeerStyle))
X = [feature(d) for d in data]
y = [d for d in data if d['beer/style'] == 'Flanders Red Ale']
print(y[0])
print(y[1])
print(y.__len__())

def featureAddition(datum):
    feat = [1]
    if(datum['beer/style'] == 'American IPA'):
        feat.append(1)
    else:
        feat.append(0)
    return feat

X = [featureAddition(d) for d in data]
Y = [d['review/taste'] for d in data]

theta, residuals, rank, s = np.linalg.lstsq(X, Y)
m, c = np.linalg.lstsq(X, Y)[0]
print(m)
print(c)

print(theta)

X1 = np.array([featureAddition(d) for d in data])
plt.plot(X, Y, 'o', label='Original data', markersize=10)
plt.plot(X1, m*X1 + c, 'r', label='Fitted line')
plt.legend()
#plt.show()

Xtrain, Xtest = np.array_split(X, 2)
print(Xtrain.__len__())
print(Xtest.__len__())
Ytrain, Ytest = np.array_split(Y, 2)

theta = np.linalg.lstsq(Xtrain, Ytrain)[0]
print(theta)

def MSE(theta, X, y):
  theta = np.matrix(theta).T
  X = np.matrix(X)
  y = np.matrix(y).T
  diff = X*theta - y
  diffSq = diff.T*diff
  diffSqReg = diffSq / len(X)
  print("offset =", diffSqReg.flatten().tolist())
  return diffSqReg.flatten().tolist()[0]

mseTrain = MSE(theta, Xtrain, Ytrain)
print("Mse Training Error: ",mseTrain)
mseTest = MSE(theta, Xtest, Ytest)
print("Mse Test Error: ", mseTest)

FamousBeerStyles = []
BeerStyleReviewCount = Counter(BeerStyle)
for key, value in BeerStyleReviewCount.items():
    if value>=50:
        FamousBeerStyles.append(key)

print(FamousBeerStyles[1])
print(FamousBeerStyles.__len__())
print(BeerStyleReviewCount.__len__())

def featureAddition1(datum):
    feat = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    beerStyle = datum['beer/style']
    if(beerStyle in FamousBeerStyles):
        index = FamousBeerStyles.index(beerStyle)
        feat[index+1] = 1
    return feat

trainData, testData = np.array_split(data, 2)
XtrainMulti = [featureAddition1(d) for d in trainData]
XtestMulti = [featureAddition1(d) for d  in testData]
print(data[1])
print(X[1])

theta = np.linalg.lstsq(XtrainMulti, Ytrain)[0]
print(theta)

mseTrain = MSE(theta, XtrainMulti, Ytrain)
print("Mse Training Error: ",mseTrain)
mseTest = MSE(theta, XtestMulti, Ytest)
print("Mse Test Error: ", mseTest)


#test_predictions = clf.predict(X_test)



