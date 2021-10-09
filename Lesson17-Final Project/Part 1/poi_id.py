#!/usr/bin/python

import sys
import pickle

import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import tester

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary']

initial_features_list = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances', 
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
'director_fees', 'shared_receipt_with_poi','to_messages','from_messages', 'from_poi_to_this_person','from_this_person_to_poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
#print(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf_GNB = GaussianNB()

from sklearn.svm import SVC
#clf = SVC(kernel='rbf',C=100)
clf_SVC = SVC()

from sklearn.tree import DecisionTreeClassifier
clf_DT= DecisionTreeClassifier()

from sklearn.cluster import KMeans
clf_Kmeans = KMeans(n_clusters=2)
#print(clf_Kmeans.get_params())

### Task 5: Tune your classifier to achieve better than .3 precision and recall 

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can check your results.

print("GaussianNB:")
tester.dump_classifier_and_data(clf_GNB, my_dataset, features_list)
tester.main()
print("SVC:")
tester.dump_classifier_and_data(clf_SVC, my_dataset, features_list)
tester.main()
print("Decision Tree:")
tester.dump_classifier_and_data(clf_DT, my_dataset, features_list)
tester.main()
print("Kmeans")
tester.dump_classifier_and_data(clf_Kmeans, my_dataset, features_list)
tester.main()

import pandas as pd
df = pd.DataFrame.from_dict(my_dataset, orient='index') # Names as index

print ('Total no of records: {}'.format(df.shape[0]))
print ('Total no of POIs: {}'.format(df.loc[df['poi'] == True].shape[0]))
print ('Total no of POIs: {}'.format(df.loc[df['poi'] == False].shape[0]))
