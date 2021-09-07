#!/usr/bin/python3

import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

#visualization is one of the most powerful tools for finding outliers!
import pandas as pd
df = pd.DataFrame(data_dict)
df.loc['salary',:] = pd.to_numeric(df.loc['salary',:], errors='coerce')

print (df.loc['salary',:].max())

data_dict.pop('TOTAL', 0)

data = featureFormat(data_dict, features)

plt.scatter(data[:,0], data[:,1])
plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

print ([name for name in df.columns if df.loc['salary', name] > 10**6 and df.loc['bonus',name] > 5*10**6])