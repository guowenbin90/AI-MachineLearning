#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from collections import Counter

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
clf = SVC(kernel='linear', C=1)

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]

t0 = time()
clf.fit(features_train,labels_train)
print("Training Time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

print("Accuracy:", accuracy_score(labels_test,pred))

print ("Predictions:")
print ("10:", pred[10])
print ("26:", pred[26])
print ("50:", pred[50])

c = Counter(pred)
print ("No of predictions for Chris(1):", c[1])
print('Number of events predicted in Chris class is', sum(clf.predict(features_test) ==1))
#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
