#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt

import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import tester

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary']

features_list = ['poi', 'deferred_income', 'total_stock_value', 'expenses', 'poi_mail_ratio']

initial_features_list = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances', 
'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
'director_fees', 'shared_receipt_with_poi','to_messages','from_messages', 'from_poi_to_this_person','from_this_person_to_poi']
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

num_samples = len(data_dict)
print ("data set length:", num_samples)

### Task 2: Remove outliers
# First remove non-people entries
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")



# Iterate over dataset and use ad hoc filters to remove other outliers, if necessary
salaryList = []
clean_dict = {}
for name, pdata in data_dict.items():
    #print (pdata["from_messages"])
    # Remove email freaks
    if pdata["from_messages"]!="NaN":
        if pdata["from_messages"]> 1000:
            continue
    clean_dict[name] = pdata
    #print(clean_dict)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

#Lists used for visualization
from_all=[]
from_poi=[]
m_ratio_list=[]

for name, pdata in clean_dict.items():

    pdata["poi_mail_ratio"] =0
    if pdata['from_messages'] != "NaN":
        from_all.append(float(pdata['from_messages']))
        from_poi.append(float(pdata['from_poi_to_this_person']))
        m_ratio = float(pdata['from_poi_to_this_person']) / float(pdata['from_messages'])
        pdata["poi_mail_ratio"] = m_ratio
        m_ratio_list.append(m_ratio)

#print ("m ratios", m_ratio_list)

plt.scatter(from_all, from_poi)
plt.xlabel("all emails")
plt.ylabel("from POI")


my_dataset = clean_dict
#print(my_dataset)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

import pandas as pd
df = pd.DataFrame.from_dict(my_dataset, orient='index') # Names as index



### Task 4: Try a varity of classifiers
# Switch to decision tree
from sklearn.tree import DecisionTreeClassifier
clf_DT= DecisionTreeClassifier(min_samples_split=6)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
'''
from time import time
t0 = time()
clf_DT.fit(features_train, labels_train)
print ("training time:", round(time()-t0, 3), "s")
'''

### Task 6: Dump your classifier, dataset, and features_list so anyone can check your results.
'''
from sklearn.model_selection import GridSearchCV
param_grid = {'min_samples_split': np.arange(2, 10)}
tree = GridSearchCV(DecisionTreeClassifier(), param_grid)
tree.fit(features_train, labels_train)
print(tree.best_params_) # get {'min_samples_split': 6}
'''

from sklearn.feature_selection import RFECV

rfecv = RFECV(estimator=clf_DT, step=1, scoring="f1")
rfecv.fit(features_train, labels_train)


print ("optimal number of features",rfecv.n_features_)
print ("Now Printing Features Priority")
for i in range(0,len(rfecv.ranking_)): 
    print (features_list[i+1], rfecv.ranking_[i])
print (rfecv.ranking_)
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (f1 score)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


print("Decision Tree:")
tester.dump_classifier_and_data(clf_DT, my_dataset, features_list)
tester.main()

