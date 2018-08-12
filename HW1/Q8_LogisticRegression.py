import numpy as np
import urllib.request
import scipy.optimize
from sklearn.metrics import accuracy_score
import random
from math import exp
from math import log

def getData(dataUrl):
    for line in urllib.request.urlopen(dataUrl):
        #print(line)
        yield eval(line)

print("Reading data")
data = list(getData("http://jmcauley.ucsd.edu/cse258/data/beer/beer_50000.json"))
print(data[0])

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  print("ll =", loglikelihood)
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0.0]*len(theta)
  product = 1
  sum1 = 0;
  for k in range(len(dl)):
      for i in range(len(X)):
          logit = inner(X[i], theta)
          sum1 += X[i][k] * (1 - sigmoid(logit))
          if not y[i]:
              sum1 -= X[i][k]
      sum1-=(2*lam*theta[k])
      dl[k] = sum1
  # Negate the return value since we're doing gradient *ascent*
  return np.array([-x for x in dl])

trainData, testData = np.array_split(data, 2)

def featureSVM(datum):
    feat = []
    feat.append(datum['beer/ABV'])
    feat.append(datum['review/taste'])
    return feat

XtrainSVM = [featureSVM(d) for d in trainData]
XtestSVM = [featureSVM(d) for d in testData]

YtrainSVM = [d['beer/style'] == 'American IPA' for d in trainData]
YtestSVM = [d['beer/style'] == 'American IPA' for d in testData]

# If we wanted to split with a validation set:
#X_valid = X[len(X)/2:3*len(X)/4]
#X_test = X[3*len(X)/4:]

# Use a library function to run gradient descent (or you can implement yourself!)
theta,l,info = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(XtrainSVM[0]), fprime, args = (XtrainSVM, YtrainSVM, 1.0))
print("Final log likelihood =", -l)
print(theta)

'''YTrainPredictions = []
for d in XtrainSVM:
    if(d*theta>0):
        YTrainPredictions.append(True)
    else:
        YTrainPredictions.append(False)'''
YTrainPredictions = [inner(d, theta)>0 for d in XtrainSVM]
YTestPredictions = [inner(d, theta)>0 for d in XtestSVM]

print(YTrainPredictions[50])
print(YtrainSVM[50])

print("Train Data Accuracy: ",accuracy_score(YtrainSVM, YTrainPredictions))
print("Test Data Accuracy: ", accuracy_score(YtestSVM, YTestPredictions))

print("End of program")