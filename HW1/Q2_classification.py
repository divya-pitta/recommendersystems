import numpy as np
import urllib.request
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

'''def parseData(fname):
  for l in urllib.request.urlopen(fname):
    yield eval(l)

print("Reading data...")
data = list(parseData("http://jmcauley.ucsd.edu/cse255/data/amazon/book_images_5000.json"))
print("done")

X = [b['image_feature'] for b in data]
y = ["Children's Books" in b['categories'] for b in data]

X_train = X[:2500]
y_train = y[:2500]

print(y_train[0])

X_test = X[2500:]
y_test = y[2500:]

# Create a support vector classifier object, with regularization parameter C = 1000
clf = svm.SVC(C=1000, kernel='linear')
clf.fit(X_train, y_train)

train_predictions = clf.predict(X_train)
test_predictions = clf.predict(X_test)'''

def getData(dataUrl):
    for line in urllib.request.urlopen(dataUrl):
        #print(line)
        yield eval(line)

#Reading the data from url line by line and creating a list out of it
print("Reading data")
data = list(getData("http://jmcauley.ucsd.edu/cse258/data/beer/beer_50000.json"))
print(data[0])

trainData, testData = np.array_split(data, 2)

def featureSVM(datum):
    feat = []
    feat.append(datum['beer/ABV'])
    feat.append(datum['review/taste'])
    if('IPA' in datum['beer/name']):
        feat.append(1)
    else:
        feat.append(0)
    if('citrus' in datum['review/text']):
        feat.append(1)
    else:
        feat.append(0)
    return feat

XtrainSVM = [featureSVM(d) for d in trainData]
XtestSVM = [featureSVM(d) for d in testData]

'''scaling = MinMaxScaler(feature_range=(-1,1)).fit(XtrainSVM)
XtrainSVM = scaling.transform(XtrainSVM)
XtestSVM = scaling.transform(XtrainSVM)'''

print(XtrainSVM[0])
print(XtestSVM[0])

YtrainSVM = [d['beer/style'] == 'American IPA' for d in trainData]
YtestSVM = [d['beer/style'] == 'American IPA' for d in testData]

print(YtrainSVM[0])
print(YtestSVM[0])

clf = svm.SVC(C=100000)
clf.fit(XtrainSVM, YtrainSVM)

train_predictions = clf.predict(XtrainSVM)
test_predictions = clf.predict(XtestSVM)

print("Train Data Accuracy: ",accuracy_score(YtrainSVM, train_predictions))
print("Test Data Accuracy: ", accuracy_score(YtestSVM, test_predictions))


print(train_predictions[0])
print("End of program")